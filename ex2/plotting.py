import numpy as np
from numpy import exp
import matplotlib.pyplot as plt
from matplotlib import cm
# pip install myavi; pip install PyQt5
from mayavi import mlab
# from main import dim

plt.rcParams['mathtext.fontset'] = 'cm'
font = {'family' : 'serif', 
        'size': 20}
plt.rc('font', **font)
plt.rc('lines', lw=1)


path = "plots/"

dim=3

def plot_single(S, h, args, name):
    J, dz, B, a = args
    T = len(S)
    N = len(S[0])
    t = np.linspace(0, T*h, T)
    coo = ["x", "y"]
    col = ["g", "b"]

    fig, ax = plt.subplots(figsize=(12, 6))
    for i in range(2):
        ax.plot(t, S[:, 0,  i], label="$S_"+coo[i]+"$", color=col[i])
    ax.plot(t, S[0, 0, 0] * np.cos(t), "k", linestyle=(0, (5, 10)), label="$S_0 \cos(\omega t)$")
    ax.legend()
    ax.set_xlabel("$t$")
    ax.set_ylabel("$S$")
    ax.set_title("$ B_z = " + str(B[2]) + ",\, \\alpha = " + str(a) + ",\, d_z =  " + str(dz) + ", \, h = " +str(h) + "$")
    plt.savefig(path + name + ".pdf")


def plot_err_afh(Sx, hs, Ts, S0, args, pows, names, name):
    J, dz, B, a = args

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, pow in enumerate(pows):
        delta = np.abs(Sx[i] - S0*np.cos((Ts[i] - 1) * hs))
        ax.loglog(
            hs, delta, "x", label="$\Delta S_\\mathrm{" + names[i] + "}$",
            color=cm.hot(i/len(pows))
            )

        c = delta[-1] * hs[-1]**(-pow)
        ax.loglog(
            hs, c*hs**pow, "--", label="$c h^{}$".format(pow),
            color=cm.winter(i/len(pows))
            )

        ax.set_xlabel("$h$")
        ax.set_ylabel("$\Delta S$")
        ax.set_title("$ B_z = " + str(B[2]) + ",\, \\alpha = " + str(a) + ",\, d_z =  " + str(dz) + "$")
        ax.legend()
    plt.tight_layout()
    plt.savefig(path + name + ".pdf")


def plot_decay(S, h, args, name):
    J, dz, B, a = args
    T = len(S)
    N = len(S[0])
    t = np.linspace(0, T*h, T)
    coo = ["x", "y"]
    col = ["g", "b"]

    fig, ax = plt.subplots(figsize=(6, 5))
    for i in range(2):
        ax.plot(t, S[:, 0,  i], label="$S_"+coo[i]+"$", color=col[i])
    ax.plot(t, S[0, 0, 0]*exp(-t*a), "k--", label="$\exp(-t / \\tau)$")

    ax.legend()
    ax.set_xlabel("$t$")
    ax.set_ylabel("$S$")
    ax.set_title("$ B_z = " + str(B[2]) + ",\, \\alpha = " + str(a) + ",\, d_z =  " + str(dz) + ", \, h = " +str(h) +  "$")
    plt.tight_layout()
    plt.savefig(path + name + ".pdf")


def plot_coords(S, h, name, args):
    J, dz, B, a = args
    T = len(S)
    N = len(S[0])
    t = np.linspace(0, T*h, T)
    spins = [str(i) for i in range(N)]

    fig, ax = plt.subplots(dim, sharex=True, sharey=True, figsize=(16, 12))
    for i in range(N):
        for j in range(dim):
            ax[j].plot(t, S[:, i, j], 
            label="$S_"+spins[i]+"$",
            color=cm.viridis(i/N))


    ax[2].set_xlabel("$t$")
    ax[2].set_ylabel("$S$")
    ax[0].set_title("$S_x$")
    ax[1].set_title("$S_y$")
    ax[2].set_title("$S_z$")
    fig.suptitle("$ B_z = " + str(B[2]) + ",\, \\alpha = " + str(a) + ",\, d_z =  " + str(dz) + ", \, h = " +str(h) +  "$")
    plt.tight_layout()
    plt.savefig(path + name + ".pdf")


def plot_spins(S):
    N = len(S)
    l = N/2
    x, y, z= np.mgrid[-l:l:N*1j, 0:0:1j, 0:0:1j]
    S = S[:, :, np.newaxis, np.newaxis]
    mlab.quiver3d(x, y, z, S[:, 0], S[:, 1], S[:, 2])
    mlab.show()


def anim_spins(S, skip=1):
    T = len(S)
    N = len(S[0])
    l = N/2
    x, y, z= np.mgrid[-l:l:N*1j, 0:0:1j, 0:0:1j]
    S = S[:, :, :, np.newaxis, np.newaxis]
    mlab.plot3d(x, y, z)
    quiver = mlab.quiver3d(
        x, y, z, S[0, :, 0], S[0, :, 1], S[0, :, 2],
        mode="arrow", scalars=np.arange(N)/N, colormap="plasma",
        resolution=16)
    quiver.glyph.color_mode = "color_by_scalar"
    
    @mlab.animate(delay=10)
    def anim():
        for i in range(T//skip):
            quiver.mlab_source.u = S[i*skip, :, 0]
            quiver.mlab_source.v = S[i*skip, :, 1]
            quiver.mlab_source.w = S[i*skip, :, 2]
            yield
    
    anim()
    mlab.show()