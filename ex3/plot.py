import numpy as np
from matplotlib import pyplot as plt
from utillities import get_D, get_tz, get_var, get_mass, rms, simpson
from matplotlib import cm
from os import path, mkdir

plt.rcParams['mathtext.fontset'] = 'cm'
font = {'family' : 'serif', 
        'size': 20}
plt.rc('font', **font)
plt.rc('lines', lw=2)

dir_path = "plots/"
fact = 60 * 60 * 24


def make_dir(dir_path):
    """ recursively (!) creates the needed directories """
    if not path.isdir(dir_path):
        make_dir("/".join(dir_path.split("/")[:-2]) + "/")
        mkdir(dir_path)


def check_dir(dir_path):
    if not path.isdir(dir_path):
        make_dir(dir_path)


def save_plot(fig, ax, fname):
    check_dir(dir_path)
    plt.tight_layout()
    plt.savefig(dir_path + fname + ".pdf")
    plt.close(fig)


def plot_C(C, args, name):
    Ceq, K, Nt, Nz, a, dz, dt, kw, L, t0 = args
    fact = 60 * 60 * 24
    extent = 0, t0/fact, L, 0
    C = C[::(Nt//500+1), ::(Nz//500+1)]

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(C.T, aspect="auto", extent=extent)
    ax.set_ylabel("$z / [\mathrm{m}]$")
    ax.set_xlabel("$t / [\mathrm{days}]$")

    fig.colorbar(im)
    fig.suptitle(
        "$K_0={:.3e},\,\\alpha={:.3e},\,k_w={} $\n\
            $N_t={},\,N_z={}$".format(K[0], a, kw, Nt, Nz), 
            fontsize=18, y=0.9
            )
    fig.tight_layout()
    
    save_plot(fig, ax, name)


def plot_Cs(Cs, args):
    Ceq, K, Nt, Nz, a, dz, dt, kw, L, t0 = args
    fact = 60 * 60 * 24
    fig, ax = plt.subplots(figsize=(8, 6))

    for i, C in enumerate(Cs):
        Nz = C.shape[0]
        z = np.linspace(0, L, Nz)
        ax.plot(z, C, color=cm.viridis(i/len(Cs)))

    ax.set_xlabel("$z / [\mathrm{ m }]$")
    ax.set_xlabel("$C / [\mathrm{ days }]$")

    fig.suptitle("$K_0={:.3e},\,\\alpha={:.2f},\,k_w={}$".format(K[0], a, kw), fontsize=12)
    fig.tight_layout()
    plt.show()


def plot_D(args):
    D = get_D(args)
    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(D.todense())
    fig.colorbar(im)
    fig.tight_layout()

    plt.show()


def plot_M(C, args, name):
    Ceq, K, Nt, Nz, a, dz, dt, kw, L, t0 = args
    C = C[::(Nt//500+1)]
    t, z = get_tz(C, args)
    M = get_mass(C, args)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(t/fact, (M-M[0])/M[0])
    ax.set_xlabel("$t / [\mathrm{days}]$")
    ax.set_ylabel("$\Delta M / M_0$")
    fig.tight_layout()

    save_plot(fig, ax, name)


def plot_var(C, args, name):
    Ceq, K, Nt, Nz, a, dz, dt, kw, L, t0 = args
    C = C[::(Nt//500+1)]
    t, z = get_tz(C, args)
    t = t / fact
    var = get_var(C, args)
    lin = var[0] + 2 * K[0] * t *fact

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(t, var, label="$\sigma^2(t)$")
    ax.plot(t, lin, "--k", label="$\sigma^2(0) + 2 K t$")
    ax.plot(t, np.max(var)*np.ones_like(t), "-.k", label="$\sigma^2(\infty)$")
    ax.set_ylim(0.95*np.min(var), 1.05*np.max(var))
    ax.set_xlabel("$t / [\mathrm{days}]$")
    ax.set_ylabel("$\sigma^2$")
    plt.legend()
    fig.tight_layout()

    save_plot(fig, ax, name)


def plot_M_decay(C, args, name):
    Ceq, K, Nt, Nz, a, dz, dt, kw, L, t0 = args
    C = C[::(Nt//500+1)]
    t, z = get_tz(C, args)
    t = t/fact
    M = np.einsum("tz -> t", C)
    tau = L / kw / fact
    Bi = kw * L / np.min(K)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(t, M, label="$M(t)$")
    ax.plot(t, M[0] * np.exp(-t / tau), "--k", label="$M(0)\exp(-t/\\tau)$")
    fig.suptitle("$\mathrm{ Bi }="+"{:.3e}".format(Bi) + ",\, \\tau = {:.3f}".format(tau) + "\,\mathrm{ days }$")
    ax.set_xlabel("$t / [\mathrm{days}]$")
    ax.set_ylabel("$M/[\mathrm{ mol/m^2 }]$")
    ax.legend()
    fig.tight_layout()

    save_plot(fig, ax, name)



def plot_minmax(C, args, name):
    Ceq, K, Nt, Nz, a, dz, dt, kw, L, t0 = args
    C = C[::(Nt//500+1), ::(Nz//500+1)]
    t, z = get_tz(C, args)
    t = t/fact
    Min = C.min(axis=1)
    Max = C.max(axis=1)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(t, Min, label="$\mathrm{ min_z } C(t)$")
    ax.plot(t, Max, label="$\mathrm{ max_z } C(t)$")
    fig.suptitle(
    "$K_0={},\,\\alpha={:.2f},\,k_w={} ".format(K[0], a, kw)\
        +",\,C_\mathrm{ eq }"+"= {}$".format(Ceq[0]), 
        fontsize=18, y=0.9
        )
    ax.set_xlabel("$t / [\mathrm{days}]$")
    ax.set_ylabel("$C / [\mathrm{ mol/m^3 }]$")
    plt.legend()

    save_plot(fig, ax, name)


def plot_conv_t(Cs, Nts, exp, args, name):
    Ceq, K, Nt, Nz, a, dz, dt, kw, L, t0 = args

    fig, ax = plt.subplots(figsize=(8, 5))
    dts = t0/fact / Nts
    errs = [rms(Cs[i], Cs[-1]) for i in range(len(Cs)-1)]

    ax.loglog(
        dts[:-1],  errs[-1]*(dts[:-1]/dts[-2])**exp, 
        label="$C \Delta t^{}$".format(exp)
        )
    ax.loglog(dts[:-1], errs, "kx", label="$\Delta_\mathrm{ rms }$")

    ax.set_xlabel("$\Delta t / [\mathrm{ days }]$")
    ax.set_ylabel("$\mathrm{rel. err.}$")
    plt.legend()
    fig.suptitle("$K_0={:.3e},\,k_w={},\,N_z={}$".format(K[0], kw, Nz), y=0.9)
    save_plot(fig, ax, name)


def plot_conv_z(Cs, Nzs, exp, args, name):
    Ceq, K, Nt, Nz, a, dz, dt, kw, L, t0 = args

    fig, ax = plt.subplots(figsize=(8, 5))
    dzs = 1 / Nzs
    # skips = [int((Nzs[-1] - 1)/(Nzs[i] - 1)) for i in range(len(Nzs))]
    # errs = [rms(Cs[i], Cs[-1][::skips[i]]) for i in range(len(Cs)-1)]

    errs = []
    z = np.linspace(0, L, Nzs[-1])
    M10 = simpson(Cs[-1] * z, x=z)
    for i, Nz in enumerate(Nzs[:-1]):
        z = np.linspace(0, L, Nz)
        M1 = simpson(Cs[i] * z, x=z)
        errs.append(np.abs(M1 - M10)/M10)

    ax.loglog(
        dzs[:-1],  errs[-1]*(dzs[:-1]/dzs[-2])**exp, 
        label="$C \Delta z^{}$".format(exp)
        )
    ax.loglog(dzs[:-1], errs, "kx", label="$\Delta_\mathrm{ rms }$")

    ax.set_xlabel("$\Delta z / [\mathrm{ L }]$")
    ax.set_ylabel("$\mathrm{rel. err.}$")
    plt.legend()
    fig.suptitle("$K_0={:.3e},\,k_w={},\,N_t={}$".format(K[0], kw, Nt), y=0.9)

    save_plot(fig, ax, name)
