import numpy as np
"""Functions Used"""


# Energy Function
def Epot(x, y, gamma=0.01):
    """
    Function to quickly calculate the energy of the given X, Y
    """
    z = 5 - 0.7 * np.log(
        (np.exp(-(x + 2) ** 2 - (y + 2) ** 2) / gamma) + (np.exp(-(x - 2) ** 2 - (y - 1) ** 2) / gamma) + (
                    np.exp(-(x + 3) ** 2 - 5 * (y - 2) ** 2) / gamma) + 0.003 * np.exp(
            (x) ** 2 - (y + 0.5) ** 2) / gamma + 0.04 * np.exp(-(x + 0.1) ** 2 - (y - 2) ** 2) / gamma)
    return z


# Generate data on the potential
def generate_data(n_windows, Nstep, T, force, last_position, restraint_position, sigma=1, gamma=0.01):
    """
    Function to generate data for the given number of images (Nimage), steps (Nstep)

    """
    beta = beta = 1 / (0.001987204259 * T)
    # Initialize the data array
    data = np.zeros((n_windows, Nstep, 2), dtype=np.float32)
    bias = np.sum(0.5 * force * (last_position - restraint_position) ** 2, axis=0)
    Elast = Epot(last_position[0], last_position[1]) + bias

    for j in range(Nstep):
        for i in range(n_windows):
            r = np.random.rand()
            b = np.random.rand()
            u = -np.log(r)
            rho = sigma * np.sqrt(2.0 * u)
            theta = 2.0 * np.pi * b
            x = rho * np.cos(theta) + last_position[0, i]
            y = rho * np.sin(theta) + last_position[1, i]
            bias = 0.5 * force[0, i] * ((x - restraint_position[0, i]) ** 2) + \
                   0.5 * force[1, i] * ((y - restraint_position[1, i]) ** 2)
            Ecurrent = bias + Epot(x, y, gamma)

            if x < -3 or x > 3 or y < -3 or y > 3:
                Ecurrent = 100000000

            fact = np.exp(-(Ecurrent - Elast[i]) * beta)
            rr = np.random.rand(1, 1)

            if rr < fact:
                last_position[0, i] = x
                last_position[1, i] = y
                Elast[i] = Ecurrent
                data[i, j, 0] = x
                data[i, j, 1] = y
            else:
                data[i, j] = last_position[:, i]

    return data, last_position


def optimize_string(data_avg):
    """
    Function to optimize the string data
    """
    ndim = data_avg.shape[1]
    n = data_avg.shape[0]
    Nnew = n

    X = np.arange(1, n + 0.1, 0.1)
    norder = 6

    p = []
    for k in range(ndim):
        p.append(np.polyfit(np.arange(1, n + 1), data_avg[:, k], deg=norder))

    Y = np.zeros(len(X))
    for j in range(ndim):
        Y = Y + (np.polyval(np.polyder(p[j]), X)) ** 2

    Y = np.sqrt(Y)
    L = np.trapezoid(Y, X)  # Careful with this, it is counterintuitive
    Li = np.linspace(0, L, Nnew)

    flen = np.zeros(len(X))
    for ibig in range(1, len(X)):
        flen[ibig] = np.trapezoid(Y[:ibig + 1], X[:ibig + 1])  # Careful with this, it is counterintuitive

    pt = np.zeros(Nnew).astype(int)
    for i in range(Nnew):
        idx = np.argmin(abs(flen - Li[i]))
        pt[i] = int(idx)

    G = np.zeros((ndim, len(X)))
    newconstr = []
    for j in range(ndim):
        G[j] = np.polyval(p[j], X)
        newconstr.append(G[j, pt])
    newconstr = np.array(newconstr)
    return newconstr
