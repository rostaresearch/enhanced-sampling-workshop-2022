import copy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.interpolate import CubicSpline


def plot_act_data(data):
    colormap = cm.get_cmap('tab20b', data.shape[2])
    f, ax = plt.subplots()
    for i in range(data.shape[0]):
        xx = np.linspace(i, i+1, data.shape[1])
        for j in range(data.shape[2]):
            if i == 0:
                ax.plot(xx, data[i, :, j], color=colormap.colors[j],
                        linewidth=0.5, label="r{:d}".format(j+1))
            else:
                ax.plot(xx, data[i, :, j], color=colormap.colors[j],
                        linewidth=0.5)
    plt.xlabel("String Points", fontsize=15)
    plt.ylabel("Distance", fontsize=15)
    ax.tick_params(labelsize=12)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left',
               borderaxespad=0.)
    plt.show()
    return


def opt_string(data, order="c", new_nw=None, write=False, plot=False):
    """

    :param data: shaped as n_windows, n_frames, n_coordinates
    :param order: c is for cubic spline fitting, integer is for polynomial
    :param new_nw:
    :param write:
    :param plot:
    :return:
    """
    nw, n_frames, nc = data.shape
    avg = np.mean(data, axis=1)
    if plot:
        plot_act_data(data)
    if new_nw is None:
        new_nw = nw
    x = np.arange(1.0, new_nw+0.1, 0.1)
    y = np.zeros(shape=x.shape, dtype=np.float_)
    if order == "c":
        p = np.empty(0)
        for i in range(nc):
            p = np.append(p, CubicSpline(np.linspace(1, new_nw, nw), avg[:, i]))
            y += np.square(p[i](x, 1))
    else:
        p = np.empty(shape=(nc, order+1), dtype=np.float_)
        for i in range(nc):
            p[i] = np.polyfit(np.linspace(1, new_nw, nw), avg[:, i], order)
            y += np.square(np.polyval(np.polyder(p[i]), x))
    y = np.sqrt(y)
    l = np.trapz(y, x)
    # l = simps(y, x)
    li = np.linspace(0, l, new_nw)
    flen = np.empty(shape=x.shape, dtype=np.float_)
    for i in range(flen.shape[0]):
        flen[i] = np.trapz(y[:i+1], x[:i+1])
    pt = np.empty(shape=new_nw, dtype=np.int_)
    for i in range(new_nw):
        v = np.abs(flen-li[i])
        pt[i] = np.where(v == np.min(v))[0]
    g = np.empty(shape=(nc, x.shape[0]), dtype=np.float_)
    newconstr = np.empty(shape=(nc, new_nw), dtype=np.float_)
    for i in range(nc):
        if order == "c":
            g[i] = p[i](x)
        else:
            g[i] = np.polyval(p[i], x)
        newconstr[i] = g[i, pt]
    newconstr = np.transpose(newconstr)
    if plot:
        colormap = cm.get_cmap('tab20b', nc)
        f, ax = plt.subplots()
        xx = range(1, new_nw + 1)
        for j in range(nc):
            ax.plot(xx, newconstr[:, j], color=colormap.colors[j], label="r{:d}".format(j + 1))
        plt.xlabel("String Windows", fontsize=15)
        plt.ylabel("Distance", fontsize=15)
        ax.tick_params(labelsize=12)
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left',
                   borderaxespad=0.)
        plt.show()
    if write:
        np.savetxt("pyconstr.dat", np.transpose(newconstr), fmt="%8.4f")
    return newconstr


def opt_nw(data2d, order="c", accept=2.5):
    nw = 10
    while True:
        skip = data2d.shape[0] % nw
        data3d = data2d[:data2d.shape[0] - skip, :].reshape((nw, int(data2d.shape[0] / nw), data2d.shape[1]))
        nc = opt_string(data3d, order=order)
        maxjump = np.max(np.abs(nc[1:] - nc[:nw-1]))
        jump = np.abs(nc[1:] - nc[:nw - 1])
        fluct = np.std(data3d, axis=1)
        overlap = np.empty((nw - 1, data2d.shape[1]))
        for i in range(nw - 1):
            overlap[i] = 1.5 * (fluct[i] + fluct[i + 1])
        if maxjump <= accept:
            break
        if np.max(jump) < 2 * np.mean(fluct):
            break
        if not np.any(overlap - jump < 0):
            break
        nw += 1
    return nc


if __name__ == "__main__":
    nw = 25
    data = []
    for i in range(1, nw + 1):
        data.append(np.genfromtxt("{0:d}/{0:d}_prod4.colvars.traj".format(i))[:, 1:])
    data = np.array(data)
    newconstr = opt_string(data, plot=True, write=True)
    with open("string.col", "r") as fin:
        template = ''.join(fin.readlines())
    for i in range(newconstr.shape[0]):         # nw
        colvar = copy.deepcopy(template)
        for j in range(newconstr.shape[1]):     # nc
            colvar = colvar.replace("X{:d}".format(j + 1), "{:.4f}".format(newconstr[i, j]))
        with open("{0:d}/string_{0:d}.col".format(i + 1), "w") as fout:
            fout.write(colvar)
    pass
