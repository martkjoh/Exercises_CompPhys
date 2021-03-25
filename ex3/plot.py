import numpy as np
from matplotlib import pyplot as plt
from utillities import get_D, get_tz, get_var, get_mass

plt.rcParams['mathtext.fontset'] = 'cm'
font = {'family' : 'serif', 
        'size': 20}
plt.rc('font', **font)
plt.rc('lines', lw=2)


def plot_C(C, args):
    Ceq, K, T, N, a, dz, kw = args
    dt = a * dz**2 * 2
    extent = 0, T*dt, 0, N*dz
    C = C[::(T//500+1), ::(N//500+1)]
    
    fig, ax = plt.subplots(figsize=(16, 10))
    im = ax.imshow(C.T, aspect="auto", extent=extent)
    fig.colorbar(im)
    fig.suptitle("$K_0={},\,\\alpha={:.2f},\,k_w={}$".format(K[0], a, kw))
    fig.tight_layout()
    plt.show()


def plot_D(args):
    D = get_D(args)
    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(D.todense())
    fig.colorbar(im)
    fig.tight_layout()

    plt.show()


def plot_M(C, args):
    Ceq, K, T, N, a, dz, kw = args
    C = C[::(T//500+1), ::(N//500+1)]
    t, z = get_tz(C, args)
    M = get_mass(C, args)

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(t, M)
    ax.set_xlabel("$t$")
    ax.set_ylabel("$M$")
    fig.tight_layout()

    plt.show()


def plot_var(C, args):
    Ceq, K, T, N, a, dz, kw = args
    dt = a * dz**2 * 2
    L, t0 = N*dz, T*dt
    C = C[::(T//500+1), ::(N//500+1)]
    t, z = get_tz(C, args)
    var = get_var(C, args)

    m = np.max(var)
    lin = var[0] + K[0] * t / 2
    i = lin<m

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(t, var)
    ax.plot(t[i], lin[i], "--k")
    ax.set_xlabel("$t$")
    ax.set_ylabel("$\sigma^2$")
    fig.tight_layout()

    plt.show()


def plot_M_decay(C, args):
    Ceq, K, T, N, a, dz, kw = args
    dt = a * dz**2 * 2
    L, t0 = N*dz, T*dt
    C = C[::(T//500+1), ::(N//500+1)]
    T, N = len(C), len(C[0])
    t = np.linspace(0, t0, T)
    z = np.linspace(0, L, N)

    M = np.einsum("tz -> t", C)
    tau = L / kw * 2
    Bi = kw * L / np.min(K)
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(t, M)
    ax.plot(t, M[0] * np.exp(-t / tau), "--k")
    ax.set_title("$\mathrm{Bi}=" + str(Bi) + ",\, \\tau = " + str(tau) + "$")
    fig.tight_layout()

    plt.show()



def plot_minmax(C, args):
    Ceq, K, T, N, a, dz, kw = args
    dt = a * dz**2 * 2
    L, t0 = N*dz, T*dt
    C = C[::(T//500+1), ::(N//500+1)]
    T, N = len(C), len(C[0])
    t = np.linspace(0, t0, T)
    z = np.linspace(0, L, N)

    Min = C.min(axis=1)
    Max = C.max(axis=1)

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(t, Min)
    ax.plot(t, Max)
    ax.set_title("$C_\mathrm{eq}"+" = {0}$".format(Ceq))

    plt.show()
