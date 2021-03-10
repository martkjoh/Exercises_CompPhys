import numpy as np
from numpy import exp
import matplotlib.pyplot as plt
from matplotlib import cm
# pip install myavi; pip install PyQt5
from mayavi import mlab
from scipy.optimize import curve_fit
from matplotlib.colors import LogNorm
import ffmpeg
import os
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
    col = ["limegreen", "royalblue"]

    fig, ax = plt.subplots(figsize=(12, 6))
    for i in range(2):
        ax.plot(t, S[:, 0,  i], label="$S_"+coo[i]+"$", color=col[i])
    ax.plot(t, S[0, 0, 0] * np.cos(t), "k", linestyle=(0, (8, 15)), lw=3, label="$S_0 \cos(\omega t)$")
    ax.plot(t, S[0, 0, 0] * np.sin(-t), "k", linestyle=(0, (3, 5)), lw=3, label="$S_0 \sin(\omega t)$")
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


def plot_decay(Ss, alphas, h, args, name):
    J, dz, B, a = args
    T = len(Ss[0])
    N = len(Ss[0][0])
    t = np.linspace(0, T*h, T)
    coo = ["x", "y"]
    col = ["limegreen", "royalblue"]

    fig, ax = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    for k, S in enumerate(Ss):
        a = alphas[k]
        for i in range(2):
            ax[k].plot(t, S[:, 0,  i], label="$S_"+coo[i]+"$", color=col[i])
        ax[k].plot(t, S[0, 0, 0]*exp(-t*a), "k--", label="$\exp(-t / \\tau)$")
        ax[k].legend()
        ax[k].set_title("$ B_z = " + str(B[2]) + ",\, \\alpha = " + str(a) + "$")
        ax[k].set_xlabel("$t$")
    ax[0].set_ylabel("$S$")
    plt.tight_layout()
    plt.savefig(path + name + ".pdf")


def plot_zs(S, h, name, args):
    J, dz, B, a = args
    T = len(S)
    N = len(S[0])
    t = np.linspace(0, T*h, T)
    spins = [str(i) for i in range(N)]

    fig, ax = plt.subplots(sharex=True, sharey=True, figsize=(8, 5))
    for i in range(N):
        ax.plot(t, S[:, i, 2], 
        label="$S_"+spins[i]+"$",
        color=cm.viridis(i/N))

        
    ax.set_xlabel("$t$")
    ax.set_ylabel("$S$")
    fig.suptitle("$ J = " + str(J) +  "$")

    plt.tight_layout()
    plt.savefig(path + name + ".pdf")


def plot_coords(S, h, name, args, alpha=1, coords=(0, 1, 2)):
    J, dz, B, a = args
    T = len(S)
    N = len(S[0])
    t = np.linspace(0, T*h, T)

    fig, ax = plt.subplots(len(coords), sharex=True, figsize=(12, 2*len(coords) + 4))
    if len(coords) == 1: ax = [ax, ]
    for i in range(N):
        for n, j in enumerate(coords):
            ax[n].plot(t, S[:, i, j],
            color=cm.viridis(i/N), alpha=alpha
            )
    
    ax[-1].set_xlabel("$t$")
    ax[-1].set_ylabel("$S$")
    coord_name=["x", "y", "z"]
    for n, i in enumerate(coords):
        ax[n].set_title("$S_" + coord_name[i] + "$")
    fig.suptitle("$ \\alpha = " + str(a) + ",\, d_z =  " + str(dz) + ",\, J = " + str(J) +  "$")

    plt.tight_layout()
    plt.savefig(path + name + ".pdf")


def plot_fit_to_sum(S, h, args, name):
    J, dz, B, a = args
    T = len(S)
    N = len(S[0])
    t = np.linspace(0, T*h, T)

    fig, ax = plt.subplots(figsize=(10, 6))

    fig.suptitle("$ \\alpha = " + str(a) + ",\, d_z =  " + str(dz) + ",\, J = " + str(J) +  "$")
    ax.set_xlabel("$t$")
    ax.set_ylabel("$S$")
    ax.set_ylim(np.min(S[:, :, 0]), np.max(S[:, :, 0]))

    def f(x, a, w, t):
        return a * np.cos(w * x) * np.exp(-x/t)

    def sci(x):
        a = "{:.2e}".format(x).split("e")
        return a[0] + "\cdot 10^{" + str(int(a[1])) + "}"

    Ssum1 = np.einsum("tnx -> tx", S)[:, 0]/N
    # Can you figure this out?
    Ssum = [Ssum1[i] for i in range(len(Ssum1)) if i%5==0 or i%11==0]
    t2 = [t[i] for i in range(len(t)) if i%5==0 or i%11==0]

    (a, w, tau), _ = curve_fit(f, t2, Ssum, maxfev=int(1e5))
    ax.plot(t, Ssum1, color="royalblue", label="$\Sigma_i S_{i, x}$")
    ax.plot(
        t, f(t, a, w, tau), "k--", 
        label="$"+sci(a)+"\cos("+sci(w)+"t)\exp(-t/"+sci(tau)+")$"
        )
    ax.legend()

    plt.tight_layout()
    plt.savefig(path + name + ".pdf")


def plot_mag(Mz, h, name, args):
    J, dz, B, a = args
    T = len(Mz)
    t = np.linspace(0, T*h, T)

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(t, Mz, label="$M_z$")
    ax.plot(t, 0*np.ones_like(t), "k--")
    ax.legend()
    ax.set_ylabel("$M$")
    ax.set_xlabel("$t$")

    fig.suptitle("$ \\alpha = " + str(a) + ",\, d_z =  " + str(dz) + ",\, J = " + str(J) +  "$")

    plt.tight_layout()
    plt.savefig(path + name + ".pdf")

def plot_spins(S, name):
    N = len(S)
    l = N/2
    mlab.figure(
        size = (1200, 600),
        bgcolor = (1,1,1), 
        fgcolor = (0.5, 0.5, 0.5)
        )
    x, y, z= np.mgrid[-l:l:N*1j, 0:0:1j, 0:0:1j]
    mlab.plot3d(x, y, z)
    S = S[:, :, np.newaxis, np.newaxis]
    quiver = mlab.quiver3d(
        x, y, z, S[:, 0], S[:, 1], S[:, 2],
        mode="arrow", scalars=np.arange(N)/N, colormap="plasma",
        resolution=16)
    quiver.glyph.color_mode = "color_by_scalar"

    mlab.roll(240)
    mlab.view(azimuth=90)
    mlab.orientation_axes()

    # I am not able to get the figuresize right w/o doing it manually
    # mlab.savefig(path + name + ".png", magnification=4, size=(1200, 800))
    mlab.show()


def anim_spins(S, name, skip=1):
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

    pad = len(str(N))
    @mlab.animate(delay=10)
    def anim():
        for i in range(T//skip):
            quiver.mlab_source.u = S[i*skip, :, 0]
            quiver.mlab_source.v = S[i*skip, :, 1]
            quiver.mlab_source.w = S[i*skip, :, 2]

            zeros = '0'*(pad - len(str(i)))
            filename = path + name + "_{}{}".format(zeros, i) + ".png"
            mlab.savefig(filename)
            yield
    
    anim()
    mlab.show()

    input = ffmpeg.input(path + name + "_%0" + str(len(str(N))) + "d.png")
    output = path + name + ".mp4"
    stream = ffmpeg.output(input, output, framerate=20)

    if os.path.isfile(output): os.remove(output)
    ffmpeg.run(stream)  

    [os.remove(path + f) for f in os.listdir(path) \
        if f.endswith(".png") and f[:len(name)]==name]