import numpy as np
import matplotlib.pyplot as plt
from deterministic_SIR import get_Nt

plt.rcParams['mathtext.fontset'] = 'cm'
font = {'family' : 'serif', 
        'size': 20}
plt.rc('font', **font)
plt.rc('lines', lw=2)


def plotSIR(x, T, dt, args, fs=(12, 8)):
    fig, ax = plt.subplots(figsize=fs)
    Nt = get_Nt(T, dt)
    t = np.linspace(0, T, Nt)
    labels = ["$S$", "$I$", "$R$"]
    for i in range(x.shape[1]):
        ax.plot(t, x[:, i], label=labels[i])
    ax.legend()
    ax.set_title("$\Delta t = {:.3e}$".format(dt))

    plt.show()
