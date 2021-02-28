import numpy as np
from numpy import exp
import matplotlib.pyplot as plt
from matplotlib import cm
# pip install myavi; pip install PyQt5
from mayavi import mlab
# from main import dim
dim=3

def plot_decay(S, h, args):
    a = args[3]
    T = len(S)
    N = len(S[0])
    t = np.linspace(0, T*h, T)
    coo = ["x", "y", "z"]

    fig, ax = plt.subplots()
    for i in range(dim):
        ax.plot(t, S[:, 0,  i], label="$S_"+coo[i]+"$")
    ax.plot(t, S[0, 0, 0]*exp(-t*a), "--")

    ax.legend()
    plt.show()


def plot_coords(S, h):
    T = len(S)
    N = len(S[0])
    t = np.linspace(0, T*h, T)
    spins = [str(i) for i in range(N)]
    fig, ax = plt.subplots(dim)
    for i in range(N):
        for j in range(dim):
            ax[j].plot(t, S[:, i, j], 
            label="$S_"+spins[i]+"$",
            color=cm.viridis(i/N),
            alpha=0.5)

    plt.show()


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
    print(S.shape)
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