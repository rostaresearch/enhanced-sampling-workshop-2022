import os
import numpy as np
import numba
import jax
import matplotlib.pyplot as plt
from time import time


def sum_Fx(expFx_old, sim):
    denom = np.sum(expFx_old * sim)
    if denom > 0:
        denom = 1 / denom
    else:
        denom = 0
    return denom * sim


def mult_3d_1d(A, B):
    C = np.empty(shape=A.shape)
    m, n, o = A.shape
    for i in range(m):
        for j in range(n):
            for k in range(o):
                C[i, j, k] = A[i, j, k] * B[k]
    return C


numba_mult = numba.jit(numba.float64[:, :, :](numba.float64[:, :, :], numba.float64[:]), nopython=True)(mult_3d_1d)


@jax.jit
def jax_mult(A, B):
    return A * B


@jax.jit
def jax_mult_broadcast(A, B):
    return A * B


def create_bins(q, numbins):
    v_min = np.min(q)
    v_max = np.max(q)
    return np.linspace(v_min, v_max, numbins)


class WHAM:
    skip = 10
    KbT = 0.001987204259 * 300  # energy unit: kcal/mol
    data = None
    k_val = None
    constr_val = None
    winsize = None
    UB = None
    Fprog = None

    def __init__(self, path):
        self.path = path
        return

    def read(self, strings=range(2, 9)):
        coor = None
        force = None
        data = None
        winsize = []
        for s in strings:
            # For string #s the constraints are in allconstr_{s-1}
            # the data is in coll_win{s}
            actcoor = np.loadtxt(os.path.join(self.path, "allconstr_{:d}.dat".format(s - 1)))
            actforce = np.loadtxt(os.path.join(self.path, "force.dat"))
            if actforce.shape != actcoor.shape:
                print("WARNING")
            winsize.append(actcoor.shape[0])
            if coor is None:
                coor = actcoor
            else:
                coor = np.append(coor, actcoor, axis=0)
            if force is None:
                force = actforce
            else:
                force = np.append(force, actforce, axis=0)
            sdata = []
            for d in range(1, winsize[-1]+1):
                u = np.loadtxt(os.path.join(self.path, "coll_win_{:d}".format(s),
                                            "data{:d}".format(d)))
                sdata.append(u[self.skip:, :])
            sdata = np.array(sdata)
            if data is None:
                data = sdata
            else:
                data = np.append(data, sdata, axis=0)
        self.data = data
        self.k_val = force
        self.constr_val = coor
        self.winsize = winsize
        return

    def calculate_UB(self):
        """
        deprecated
        :return:
        """
        numsims = self.data.shape[0]
        datlength = self.data.shape[1]

        UB = np.empty(shape=(numsims * numsims * datlength), dtype=np.float_)
        kk = 0
        for i in range(numsims):
            for j in range(datlength):
                UB[kk:kk + numsims] = np.exp(
                   -np.sum(0.5 * self.k_val[:, :] * np.square(self.constr_val[:, :] - self.data[i, j, :]),
                           axis=1) / self.KbT)
                kk += numsims
        self.UB = UB
        return

    def calculate_UB3d(self):
        numsims = self.data.shape[0]
        datlength = self.data.shape[1]

        UB = np.empty(shape=(numsims, datlength, numsims), dtype=np.float_)
        for i in range(numsims):
            for j in range(datlength):
                UB[i, j, :] = np.exp(
                   -np.sum(0.5 * self.k_val[:, :] * np.square(self.constr_val[:, :] - self.data[i, j, :]),
                           axis=1) / self.KbT)
        self.UB3d = UB
        return

    def converge(self, threshold=0.01):
        if self.UB is None:
            # self.calculate_UB()
            self.calculate_UB3d()
        numsims = self.data.shape[0]
        datlength = self.data.shape[1]
        if self.Fprog is None:
            Fprog = []
            Fx_old = np.ones(shape=numsims, dtype=np.float_)
        else:
            Fprog = self.Fprog
            Fx_old = Fprog[-1]
        change = 0.2

        while change > threshold:
            expFx_old = datlength * np.exp(Fx_old / self.KbT)
            # t1 = time()
            # a = self.UB3d * expFx_old
            # a = numba_mult(self.UB3d, expFx_old)
            a = jax_mult(self.UB3d, expFx_old)
            # t2 = time()
            sum = np.sum(a, axis=2)
            # t3 = time()
            denom = np.divide(1, sum, where=sum != 0)
            # Fxf = np.zeros(shape=self.UB3d.shape, dtype=np.float_)
            # t5 = time()
            # Fxf = self.UB3d * denom[:, :, None]
            Fxf = jax_mult_broadcast(self.UB3d, denom[:, :, None])
            # t6 = time()
            Fx = np.sum(Fxf, axis=(0, 1))
            # t7 = time()
            Fx = -self.KbT * np.log(Fx)
            Fx -= Fx[-1]
            Fx_old = Fx
            Fprog.append(Fx)
            if len(Fprog) > 1:
                change = np.max(np.abs(Fprog[-2][1:] - Fprog[-1][1:]))
            # print(t2 - t1, t3 - t2, t6 - t5, t7 - t6)
            print(change)
        self.Fprog = Fprog
        return

    def project_2d(self, cv, numbins_q=50):
        numsims = self.data.shape[0]
        datlength = self.data.shape[1]
        q1 = np.sum(self.data * cv[0], axis=2)
        # k_q1 = np.sum(self.constr_val * cv[0], axis=1)
        q2 = np.sum(self.data * cv[1], axis=2)
        # k_q2 = np.sum(self.constr_val * cv[1], axis=1)
        qep = q1 + q2
        qspace12 = create_bins(qep, numbins_q)
        qspace1 = create_bins(q1, numbins_q)
        qspace2 = create_bins(q2, numbins_q)
        Pq12 = np.zeros(shape=numbins_q, dtype=np.float_)
        Pq1 = np.zeros(shape=numbins_q, dtype=np.float_)
        Pq2 = np.zeros(shape=numbins_q, dtype=np.float_)
        Pq2d = np.zeros(shape=(numbins_q, numbins_q), dtype=np.float_)
        PepPersim = np.zeros(shape=(numsims, numbins_q), dtype=np.float_)
        for i in range(numsims):
            for j in range(datlength):
                indq = np.digitize(qep[i, j], qspace12) - 1
                indq1 = np.digitize(q1[i, j], qspace1) - 1
                indq2 = np.digitize(q2[i, j], qspace2) - 1
                Ubias = np.sum(0.5 * self.k_val[:, :] * np.square(self.constr_val[:, :] - self.data[i, j, :]), axis=1)
                denom = np.sum(datlength * np.exp((self.Fprog[-1] - Ubias) / self.KbT))
                Pq12[indq] += 1 / denom
                Pq1[indq1] += 1 / denom
                Pq2[indq2] += 1 / denom
                Pq2d[indq1, indq2] += 1 / denom
                PepPersim[i, indq] += 1 / denom
        rUep = -self.KbT * np.log(Pq12)
        valu = np.min(rUep[:int(numbins_q/2)])
        self.rUep = rUep - valu
        self.rUepPerSim = -self.KbT * np.log(PepPersim) - valu
        self.rUq2d = -self.KbT * np.log(Pq2d) - valu
        self.qspace1 = qspace1
        self.qspace2 = qspace2
        self.qspace12 = qspace12
        return

    def plot_strings(self, title):
        numsims = self.data.shape[0]
        f, a = plt.subplots()
        a.plot(self.qspace12, self.rUep, color="black")
        for i in range(numsims):
            a.plot(self.qspace12, self.rUepPerSim[i], linewidth=0.3)
        plt.title(title)
        # plt.show()
        plt.savefig()
        return
