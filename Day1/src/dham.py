import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from scipy.linalg import eig
from scipy.optimize import minimize


def rmsd(offset, a, b):
    return np.sqrt(np.mean(np.square((a + offset) - b)))


def align(query, ref):
    offset = -10.0
    res = minimize(rmsd, offset, args=(query, ref))
    print(res.x[0], res.fun)
    return res.x[0]


def count_transitions(b, numbins, lagtime, endpt=None):
    if endpt is None:
        endpt = b
    Ntr = np.zeros(shape=(b.shape[0], numbins, numbins), dtype=np.int)  # number of transitions
    for k in range(b.shape[0]):
        for i in range(lagtime, b.shape[1]):
            try:
                Ntr[k, b[k, i - lagtime] - 1, endpt[k, i] - 1] += 1
            except IndexError:
                continue
    sumtr = np.sum(Ntr, axis=0)
    trvec = np.sum(Ntr, axis=2)
    # sym = 0.5 * (sumtr + np.transpose(sumtr))
    # anti = 0.5 * (sumtr - np.transpose(sumtr))
    # print("Degree of symmetry:",
    #       (np.linalg.norm(sym) - np.linalg.norm(anti)) / (np.linalg.norm(sym) + np.linalg.norm(anti)))
    return sumtr, trvec


class DHAM:
    KbT = 0.001987204259 * 300  # energy unit: kcal/mol
    epsilon = 0.00001
    data = None
    vel = None
    datlength = None
    k_val = None
    constr_val = None
    qspace = None
    numbins = 150
    lagtime = 1

    def __init__(self):
        return

    def setup(self, dist, T, K, centres):
        self.data = dist
        self.KbT = 0.001987204259 * T
        self.k_val = np.array(K)
        self.constr_val = np.array(centres)
        return

    def build_MM(self, sumtr, trvec, biased=False):
        MM = np.empty(shape=sumtr.shape, dtype=np.float128)
        if biased:
            MM = np.zeros(shape=sumtr.shape, dtype=np.float128)
            qsp = self.qspace[1] - self.qspace[0]
            for i in range(sumtr.shape[0]):
                for j in range(sumtr.shape[1]):
                    if sumtr[i, j] > 0:
                        sump1 = 0.0
                        for k in range(trvec.shape[0]):
                            u = 0.5 * self.k_val[k] * np.square(self.constr_val[k] - self.qspace - qsp / 2) / self.KbT
                            if trvec[k, i] > 0:
                                sump1 += trvec[k, i] * np.exp(-(u[j] - u[i]) / 2)
                        MM[i, j] = sumtr[i, j] / sump1
            MM = MM / np.sum(MM, axis=1)[:, None]
        else:
            MM[:, :] = sumtr / np.sum(sumtr, axis=1)[:, None]
        return MM

    def run(self, plot=True, adjust=True, biased=False, conversion=2E-13):
        """

        :param plot:
        :param adjust:
        :param biased:
        :param conversion: from timestep to seconds
        :return:
        """
        v_min = np.nanmin(self.data) - self.epsilon
        v_max = np.nanmax(self.data) + self.epsilon
        qspace = np.linspace(v_min, v_max, self.numbins + 1)
        self.qspace = qspace
        b = np.digitize(self.data[:, :], qspace)
        sumtr, trvec = count_transitions(b, self.numbins, self.lagtime)
        MM = self.build_MM(sumtr, trvec, biased)
        d, v = eig(np.transpose(MM))
        mpeq = v[:, np.where(d == np.max(d))[0][0]]
        mpeq = mpeq / np.sum(mpeq)
        rate = np.float_(- self.lagtime * conversion / np.log(d[np.argsort(d)[-2]]))
        mU2 = - self.KbT * np.log(mpeq)
        if adjust:
            mU2 -= np.min(mU2[:int(self.numbins)])
        dG = np.max(mU2[:int(self.numbins)])
        A = rate / np.exp(- dG / self.KbT)
        x = qspace[:self.numbins] + (qspace[1] - qspace[0])
        if plot:
            plt.plot(x, mU2)
            plt.title("Lagtime={0:d} Nbins={1:d}".format(self.lagtime, self.numbins))
            plt.show()
        return x, mU2, A

    def bootstrap_error(self, size, iter=100, plotall=False, save=None):
        full = self.run(plot=False)
        results = []
        data = np.copy(self.data)
        for _ in range(iter):
            idx = np.random.randint(data.shape[0], size=size)
            self.data = data[idx, :]
            try:
                results.append(self.run(plot=False, adjust=False))
            except ValueError:
                print(idx)
        r = np.array(results).astype(np.float_)
        r = r[~np.isnan(r).any(axis=(1, 2))]
        r = r[~np.isinf(r).any(axis=(1, 2))]
        if plotall:
            f, a = plt.subplots()
            for i in range(r.shape[0]):
                a.plot(r[i, 0], r[i, 1])
            plt.show()
        # interpolation
        newU = np.empty(shape=r[:, 1, :].shape, dtype=np.float_)
        for i in range(r.shape[0]):
            newU[i, :] = np.interp(full[0], r[i, 0], r[i, 1])
            # realign
            offset = align(newU[i, :], full[1])
            newU[i, :] += offset
        stderr = np.std(newU, axis=0)
        f, a = plt.subplots()
        a.plot(full[0], full[1])
        a.fill_between(full[0], full[1] - stderr, full[1] + stderr, alpha=0.2)
        plt.title("lagtime={0:d} bins={1:d}".format(self.lagtime, self.numbins))
        if save is None:
            plt.show()
        else:
            plt.savefig(save)
        self.data = data
        return
