import os
import numpy as np
#import jax
import matplotlib.pyplot as plt


#@jax.jit
def jax_mult(A, B):
    return A * B


#@jax.jit
def jax_mult_broadcast(A, B):
    return A * B


def create_bins(q, numbins):
    v_min = np.min(q)
    v_max = np.max(q)
    return np.linspace(v_min, v_max, numbins)


def cnt_pop(projection_colvar, qspace, denom, numsims, numbins=50):
    b = np.digitize(projection_colvar, qspace) - 1
    PpS = np.empty(shape=(numsims, numbins))
    for i in range(numbins):
        md = np.ma.masked_array(denom, mask=~(b == i))
        # P[i] = np.ma.sum(md)
        PpS[:, i] = np.ma.sum(md, axis=1)
    P = np.sum(PpS, axis=0)
    return P, PpS


class WHAM:
    """
    data is 3D: (number of sims, points per sims, number of colvars)
    kval and constr_val are 2D: (number of sims, number of colvars)
    """
    skip = 10
    KbT = 0.001987204259 * 300  # energy unit: kcal/mol
    data = None
    k_val = None
    constr_val = None
    winsize = None
    UB = None
    Fprog = None
    denom = None

    def __init__(self):
        self.path = os.getcwd()
        return

    def setup(self, dist, T, K, centres):
        self.skip = int(dist.shape[1] * 0.1)
        self.KbT = 0.001987204259 * T
        if len(dist.shape) == 2:
            self.data = dist.reshape((dist.shape[0], dist.shape[1], 1))[:, self.skip:, :]
            self.k_val = np.array(K).reshape((dist.shape[0], 1))
            self.constr_val = np.array(centres).reshape((dist.shape[0], 1))
        elif len(dist.shape) == 3:
            self.data = dist
            self.k_val = np.array(K)
            self.constr_val = np.array(centres)
        else:
            raise TypeError("data is not in the right format")
        return

    def calculate_UB3d(self):
        numsims = self.data.shape[0]
        datlength = self.data.shape[1]

        UB = np.empty(shape=(numsims, datlength, numsims), dtype=np.float32)
        for i in range(numsims):
            for j in range(datlength):
                UB[i, j, :] = np.exp(
                   -np.sum(0.5 * self.k_val[:, :] * np.square(self.constr_val[:, :] - self.data[i, j, :]),
                           axis=1) / self.KbT)
        self.UB3d = UB
        return

    def converge(self, threshold=0.01):
        if self.UB is None:
            self.calculate_UB3d()
        numsims = self.data.shape[0]
        datlength = self.data.shape[1]
        if self.Fprog is None:
            Fprog = []
            Fx_old = np.ones(shape=numsims, dtype=np.float32)
        else:
            Fprog = self.Fprog
            Fx_old = Fprog[-1]
        change = 0.2

        while change > threshold:
            expFx_old = datlength * np.exp(Fx_old / self.KbT)
            a = jax_mult(self.UB3d, expFx_old)
            sum = np.sum(a, axis=2)
            denom = np.divide(1, sum, where=sum != 0)
            Fxf = jax_mult_broadcast(self.UB3d, denom[:, :, None])
            Fx = np.sum(Fxf, axis=(0, 1))
            Fx = -self.KbT * np.log(Fx)
            Fx -= Fx[-1]
            Fx_old = Fx
            if len(Fprog) > 1:
                change = np.nanmax(np.abs(Fprog[-1][1:] - Fx[1:]))
            if len(Fprog) > 2:
                prevchange = np.nanmax(np.abs(Fprog[-2][1:] - Fprog[-1][1:]))
                if prevchange < change:
                    print("The iteration started to diverge.")
                    break
            Fprog.append(Fx)
            # print(change)
        self.Fprog = Fprog
        return

    def project_1d(self, cv, numbins_q=50):
        numsims = self.data.shape[0]
        projection_colvar = np.sum(self.data * cv, axis=2)
        projection_bins = create_bins(projection_colvar, numbins_q)
        if self.denom is None:
            self.calc_denom()
        P, PpS = cnt_pop(projection_colvar, projection_bins, self.denom, numsims=numsims, numbins=numbins_q)
        profile = -self.KbT * np.log(P)
        valu = np.min(profile[:int(numbins_q/2)])
        self.profile = profile - valu
        self.profilePerSim = -self.KbT * np.log(PpS) - valu
        self.projection_bins = projection_bins
        return

    def project_2d(self, cv, numbins_q=50):
        numsims = self.data.shape[0]
        datlength = self.data.shape[1]
        colvar1 = np.sum(self.data * cv[0], axis=2)
        colvar2 = np.sum(self.data * cv[1], axis=2)
        projection_colvar = colvar1 + colvar2
        projection_bins = create_bins(projection_colvar, numbins_q)
        colvar1_bins = create_bins(colvar1, numbins_q)
        colvar2_bins = create_bins(colvar2, numbins_q)
        Pq12 = np.zeros(shape=numbins_q, dtype=np.float_)
        Pq1 = np.zeros(shape=numbins_q, dtype=np.float_)
        Pq2 = np.zeros(shape=numbins_q, dtype=np.float_)
        Pq2d = np.zeros(shape=(numbins_q, numbins_q), dtype=np.float_)
        PepPersim = np.zeros(shape=(numsims, numbins_q), dtype=np.float_)
        for i in range(numsims):
            for j in range(datlength):
                indq = np.digitize(projection_colvar[i, j], projection_bins) - 1
                indq1 = np.digitize(colvar1[i, j], colvar1_bins) - 1
                indq2 = np.digitize(colvar2[i, j], colvar2_bins) - 1
                Ubias = np.sum(0.5 * self.k_val[:, :] * np.square(self.constr_val[:, :] - self.data[i, j, :]), axis=1)
                denom = np.sum(datlength * np.exp((self.Fprog[-1] - Ubias) / self.KbT))
                Pq12[indq] += 1 / denom
                Pq1[indq1] += 1 / denom
                Pq2[indq2] += 1 / denom
                Pq2d[indq1, indq2] += 1 / denom
                PepPersim[i, indq] += 1 / denom
        profile = -self.KbT * np.log(Pq12)
        valu = np.min(profile[:int(numbins_q/2)])
        self.profile = profile - valu
        self.profilePerSim = -self.KbT * np.log(PepPersim) - valu
        self.profile2d = -self.KbT * np.log(Pq2d) - valu
        self.colvar1_bins = colvar1_bins
        self.colvar2_bins = colvar2_bins
        self.projection_bins = projection_bins
        return

    def plot_strings(self):
        numsims = self.data.shape[0]
        f, a = plt.subplots()
        a.plot(self.projection_bins, self.profile, color="black")
        for i in range(numsims):
            a.plot(self.projection_bins, self.profilePerSim[i], linewidth=0.3)
        plt.show()
        return

    def calc_denom(self):
        numsims = self.data.shape[0]
        datlength = self.data.shape[1]
        d = np.zeros(shape=(numsims, datlength))
        for i in range(numsims):
            for j in range(datlength):
                Ubias = np.sum(0.5 * self.k_val[:, :] * np.square(self.constr_val[:, :] - self.data[i, j, :]), axis=1)
                denom = np.sum(datlength * np.exp((self.Fprog[-1] - Ubias) / self.KbT))
                d[i, j] = 1 / denom
        self.denom = d
        return
