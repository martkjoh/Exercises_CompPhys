import numpy as np
from matplotlib import pyplot as plt
from utillities import get_D, get_tz, get_var, get_mass
from matplotlib import cm

plt.rcParams['mathtext.fontset'] = 'cm'
font = {'family' : 'serif', 
        'size': 20}
plt.rc('font', **font)
plt.rc('lines', lw=2)


def plot_C(C, args):
    Ceq, K, Nt, Nz, a, dz, dt, kw = args
    fact = 60 * 60 * 24
    extent = 0, Nt*dt/fact, -Nz*dz, 3
    C = C[::(Nt//500+1), ::(Nz//500+1)]

    fig, ax = plt.subplots(figsize=(16, 10))
    im = ax.imshow(C.T, aspect="auto", extent=extent)
    ax.set_ylabel("$z / [\mathrm{m}]$")
    ax.set_xlabel("$t / [\mathrm{days}]$")

    fig.colorbar(im)
    fig.suptitle("$K_0={},\,\\alpha={:.2f},\,k_w={}$".format(K[0], a, kw))
    fig.tight_layout()
    plt.show()


def plot_Cs(Cs, args):
    Ceq, K, Nt, Nz, a, dz, dt, kw = args
    fact = 60 * 60 * 24
    fig, ax = plt.subplots(figsize=(16, 10))

    for i, C in enumerate(Cs):
        z = np.linspace(0, Nz * dz, Nz)
        ax.plot(z, C, color=cm.viridis(i/len(Cs)))

    ax.set_xlabel("$z / [\mathrm{m}]$")
    ax.set_xlabel("$C / [\mathrm{??}]$")

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
    Ceq, K, Nt, Nz, a, dz, dt, kw = args
    C = C[::(Nt//500+1)]
    t, z = get_tz(C, args)
    M = get_mass(C, args)

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(t, M-M[0])
    ax.set_xlabel("$t$")
    ax.set_ylabel("$\Delta M$")
    fig.tight_layout()

    plt.show()


def plot_var(C, args):
    Ceq, K, Nt, Nz, a, dz, dt, kw = args
    C = C[::(Nt//500+1)]
    t, z = get_tz(C, args)
    var = get_var(C, args)

    m = np.max(var)
    lin = var[0] + K[0] * t / 2
    i = lin<m

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(t, var)
    ax.plot(t, lin, "--k")
    ax.set_ylim(0.95*np.min(var), 1.05*np.max(var))
    ax.set_xlabel("$t$")
    ax.set_ylabel("$\sigma^2$")
    fig.tight_layout()

    plt.show()


def plot_M_decay(C, args):
    Ceq, K, Nt, Nz, a, dz, dt, kw = args
    C = C[::(Nt//500+1)]
    t, z = get_tz(C, args)
    L = Nz * dz

    M = np.einsum("tz -> t", C)
    tau = L / kw
    Bi = kw * L / np.min(K)
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(t, M)
    ax.plot(t, M[0] * np.exp(-t / tau), "--k")
    ax.set_title("$\mathrm{Bi}=" + str(Bi) + ",\, \\tau = " + str(tau) + "$")
    fig.tight_layout()

    plt.show()



def plot_minmax(C, args):
    Ceq, K, Nt, Nz, a, dz, dt, kw = args
    C = C[::(Nt//500+1), ::(Nz//500+1)]
    t, z = get_tz(C, args)

    Min = C.min(axis=1)
    Max = C.max(axis=1)

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(t, Min)
    ax.plot(t, Max)
    ax.set_title("$C_\mathrm{eq}"+" = {0}$".format(Ceq))

    plt.show()


def plot_conv(Cs, Ns, args):
    Ceq, K, Nt, Nz, a, dz, dt, kw = args

    fac = 10 * 60 * 60 * 24
    fig, ax = plt.subplots(figsize=(12, 8))
    z = np.linspace(0, Nz*dz, Nz)
    # Mref = get_mass(Cs[-1][np.newaxis], args)
    for i in range(len(Cs) - 1):
        err = np.sqrt(np.mean((Cs[-1]-Cs[i])**2))
        ax.loglog(fac/Ns[i], err, "kx")
    
    plt.show()