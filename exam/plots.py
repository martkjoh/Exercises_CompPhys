import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from deterministic_SIR import get_Nt, get_asymptotes2
from matplotlib.colors import LogNorm
from os import path, mkdir


plt.rcParams['mathtext.fontset'] = 'cm'
font = {'family' : 'serif', 
        'size': 20}
plt.rc('font', **font)
plt.rc('lines', lw=2)

DIR_PATH="plots/"


def make_dir(dir_path):
    """ recursively (!) creates the needed directories """
    if not path.isdir(dir_path):
        make_dir("/".join(dir_path.split("/")[:-2]) + "/")
        mkdir(dir_path)


def check_dir(dir_path):
    if not path.isdir(dir_path):
        make_dir(dir_path)


def save_plot(fig, ax, fname, dir_path):
    check_dir(dir_path)
    plt.tight_layout()
    plt.savefig(dir_path + fname + ".pdf")
    plt.close(fig)


labels = ["$S$", "$I$", "$R$"]
colors = [cm.plasma(0.2), cm.plasma(0.5), cm.plasma(0.8)]

labels2 = ["$S$", "$E$", "$I$", "$I_a$", "$R$"]
colors2 = [cm.viridis(i/(len(labels2)-1)) for i in range(len(labels2))]


def plotSIR(x, T, dt, args, fs=(12, 8), name="", subdir=""):
    fig, ax = plt.subplots(figsize=fs)
    Nt = get_Nt(T, dt)
    t = np.linspace(0, T, Nt)
    N = np.sum(x[0])
    x = x / N # Normalize
    
    for i in range(x.shape[1]):
        ax.plot(t, x[:, i], label=labels[i], color=colors[i])
    S_inf, R_inf = get_asymptotes2(args)
    ax.plot(t, np.ones_like(t)*S_inf, "--", label="$S(\infty)$", color=colors[0])
    ax.plot(t, np.ones_like(t)*R_inf, "--", label="$R(\infty)$", color=colors[2])
    ax.legend()
    ax.set_title(
        "$\Delta t = {:.3e}$".format(dt)
        + "$,\,\\beta={}$".format(args[0])
        + "$,\,\\tau={}$".format(args[1])
    )


    save_plot(fig, ax, name, DIR_PATH+subdir)


def plotSIRs(result0, result, fs=(12, 8), name="", subdir=""):
    fig, ax = plt.subplots(figsize=fs)

    xs, T, dt, args = result
    Nt = get_Nt(T, dt)
    t = np.linspace(0, T, Nt)

    for x in xs:
        N = np.sum(x[0])
        x = x / N # Normalize

        for i in range(x.shape[1]):
            ax.plot(t, x[:, i], color=colors[i], alpha=0.3)

    x0, T, dt, args = result0
    Nt = get_Nt(T, dt)
    t = np.linspace(0, T, Nt)

    for i in range(x0.shape[1]):
        ax.plot(t, x0[:, i], "k--")
    S_inf, R_inf = get_asymptotes2(args)
    ax.plot(t, np.ones_like(t)*S_inf, "--", label="$S(\infty)$", color=colors[0])
    ax.plot(t, np.ones_like(t)*R_inf, "--", label="$R(\infty)$", color=colors[2])

    ax.set_title(
        "$\Delta t = {:.3e}$".format(dt)
        + "$,\,\\beta={}$".format(args[0])
        + "$,\,\\tau={}$".format(args[1])
    )

    save_plot(fig, ax, name, DIR_PATH+subdir)



def plotSEIIaRs(result0, result, fs=(12, 8), name="", subdir=""):
    fig, ax = plt.subplots(figsize=fs)

    xs, T, dt, args = result
    Nt = get_Nt(T, dt)
    save = result[0][0].shape[0]
    t = np.linspace(0, T, save)

    for x in xs:
        N = np.sum(x[0])
        x = x / N # Normalize
        for i in range(x.shape[1]):
            ax.plot(t, x[:, i], color=colors2[i], alpha=0.3)
    ax.legend([*labels2])

    x, T, dt, args = result0
    Nt = get_Nt(T, dt)
    t = np.linspace(0, T, Nt)
    N = np.sum(x[0])
    x = x / N # Normalize
    for i in range(x.shape[1]):
        ax.plot(t, x[:, i], "k--")

    save_plot(fig, ax, name, DIR_PATH+subdir)


def plotOslo(result, fs=(12, 8), name="", subdir=""):
    fig, ax = plt.subplots(figsize=fs)

    xs, T, dt, args = result
    x = xs[:, :, 0]
    Nt = get_Nt(T, dt)
    save = x.shape[0]
    t = np.linspace(0, T, save)

    N = np.sum(x[0])
    x = x / N # Normalize
    for i in range(x.shape[1]):
        ax.plot(t, x[:, i], color=colors2[i])
    ax.legend([*labels2])

    
    save_plot(fig, ax, name, DIR_PATH+subdir)


def plot_sum_inf(result, fs=(12, 8), name="", subdir=""):
    fig, ax = plt.subplots(figsize=fs)

    x, T, dt, args = result
    I_tot = x[:, 2] + x[:, 3]
    infected_cities = np.sum(I_tot>10, axis=1)
    save = x.shape[0]
    t = np.linspace(0, T, save)
    ax.plot(t, infected_cities)

    save_plot(fig, ax, name, DIR_PATH+subdir)


def plot_two_towns(result, fs=(12, 8), name="", subdir=""):
    xs, T, dt, args = result
    N_cities = xs.shape[1]
    Nt = get_Nt(T, dt)
    t = np.linspace(0, T, Nt)
    fig, ax = plt.subplots(1, 2, figsize=fs)
    for n in range(2):
        x = xs[:, :, n]
        N = np.sum(x[0])
        x = x / N # Normalize

        for i in range(x.shape[1]):
            ax[n].plot(t, x[:, i], color=colors2[i], alpha=1)

    save_plot(fig, ax, name, DIR_PATH+subdir)



def plot_many_towns(result, fs=(12, 8), name="", subdir="", shape=(2, 5)):
    xs, T, dt, args = result
    N_cities = xs.shape[1]
    Nt = get_Nt(T, dt)
    t = np.linspace(0, T, Nt)
    fig, ax = plt.subplots(*shape, figsize=fs)

    for j in range(shape[0]):
        for k in range(shape[1]):
            n = j*shape[1] + k
            x = xs[:, :, n]
            N = np.sum(x[0])
            x = x / N # Normalize

            for i in range(x.shape[1]):
                ax[j, k].plot(t, x[:, i], color=colors2[i], alpha=1)


    save_plot(fig, ax, name, DIR_PATH+subdir)



def plotIs(result, fs=(12, 8), name="", subdir=""):
    fig, ax = plt.subplots(figsize=fs)

    xs, T, dt, args = result
    N = np.sum(xs[0][0])
    for x in xs:
        x = x / N # Normalize
        Nt = get_Nt(T, dt)
        Nt0 = (Nt-1)//2 + 1
        T0 = T*((Nt0-1)/(Nt-1))
        t, dt0 = np.linspace(0, T0, Nt0, retstep=True)
        assert np.isclose(dt0, dt)

        ax.semilogy(t, x[:Nt0, 1], color=colors[1], alpha=0.3)

    a = args[0] - 1 / args[1]
    ax.semilogy(t, xs[0][0, 1]/N*np.exp(a*t), "--k", label="$\exp([\\beta -1/\\tau]t)$")
    ax.legend()
    ax.set_title(
        "$\Delta t = {:.3e}$".format(dt)
        + "$,\,\\beta={}$".format(args[0])
        + "$,\,\\tau={}$".format(args[1])
    )

    save_plot(fig, ax, name, DIR_PATH+subdir)


def plotEs(result, frac=10, fs=(12, 8), name="", subdir=""):
    fig, ax = plt.subplots(figsize=fs)

    xs, T, dt, args = result
    N = np.sum(xs[0][0])
    xs = xs / N # Normalize
    Nt = get_Nt(T, dt)
    Nt0 = (Nt-1)//frac + 1
    T0 = T*((Nt0-1)/(Nt-1))
    t, dt0 = np.linspace(0, T0, Nt0, retstep=True)
    assert np.isclose(dt0, dt)

    for i, x in enumerate(xs):
        ax.semilogy(t, x[:Nt0, 1], color=cm.viridis(i/len(xs)))

    save_plot(fig, ax, name, DIR_PATH+subdir)


def plotEsafrs(result, frac, fs=(12, 8), name="", subdir=""):
    fig, ax = plt.subplots(figsize=fs)

    xs, T, dt, args, rss, av_growth = result
    N = np.sum(xs[0][0])
    xs = xs / N # Normalize
    Nt = get_Nt(T, dt)
    Nt0 = (Nt-1)//frac + 1
    T0 = T*((Nt0-1)/(Nt-1))
    t, dt0 = np.linspace(0, T0, Nt0, retstep=True)

    assert np.isclose(dt0, dt)

    for i, x in enumerate(xs):
        ax.semilogy(t, x[:Nt0, 1], color=cm.viridis(i/len(xs)))

    # The index of the highest v with positive growth rate
    high_i = np.arange(0, len(rss))[np.greater(av_growth, 0)][-1]
    ax.semilogy(t, xs[high_i, :Nt0, 1], "k--")
    print("Corr growth rate: {}".format(av_growth[high_i]))
    print("Reach at index {} of {}".format(high_i, len(rss)))
    print("highest r_s value stillin yielding exp grwoth: {}".format(rss[high_i]))

    save_plot(fig, ax, name, DIR_PATH+subdir)


def plotEav(result, frac=10, fs=(12, 8), name="", subdir=""):
    fig, ax = plt.subplots(figsize=fs)

    xs, T, dt, args = result
    N = np.sum(xs[0][0])
    Nt = get_Nt(T, dt)
    Nt0 = (Nt-1)//frac + 1
    T0 = T*((Nt0-1)/(Nt-1))
    t, dt0 = np.linspace(0, T0, Nt0, retstep=True)
    assert np.isclose(dt0, dt)

    E_av = np.zeros(Nt0, dtype=type(xs[0]))
    for x in xs:
        x = x / N # Normalize
        E_av += x[:Nt0, 1]

    E_av *= 1/len(xs)
    ax.semilogy(t, E_av, "k--")

    save_plot(fig, ax, name, DIR_PATH+subdir)



def plotI(x, T, dt, args, fs=(12, 8), name="", subdir=""):
    fig, ax = plt.subplots(figsize=fs)
    Nt = get_Nt(T, dt)
    Nt0 = (Nt-1)//2 + 1
    T0 = T*((Nt0-1)/(Nt-1))
    t, dt0 = np.linspace(0, T0, Nt0, retstep=True)
    assert np.isclose(dt0, dt)
    N = np.sum(x[0])
    x = x / N # Normalize

    ax.semilogy(t, x[:Nt0, 1], label=labels[1], color=colors[1])
    a = args[0] - 1 / args[1]
    ax.semilogy(t, x[0, 1]*np.exp(a*t), "--k", label="$\exp([\\beta -1/\\tau]t)$")
    ax.legend()
    ax.set_title(
        "$\Delta t = {:.3e}$".format(dt)
        + "$,\,\\beta={}$".format(args[0])
        + "$,\,\\tau={}$".format(args[1])
    )

    save_plot(fig, ax, name, DIR_PATH+subdir)


def plot_maxI(max_I, betas, high_i, fs=(12, 8), name="", subdir=""):
    fig, ax = plt.subplots(figsize=fs)
    ax.plot(betas, max_I, "k.-")
    ax.plot(betas[high_i], max_I[high_i], "rx")
    ax.plot(betas, 0.2*np.ones_like(betas))
    
    save_plot(fig, ax, name, DIR_PATH+subdir)


def plot_vacc(growth_rate, vacc, high_i, fs=(12, 8), name="", subdir=""):
    fig, ax = plt.subplots(figsize=fs)
    ax.plot(vacc, growth_rate, "k.-")
    ax.plot(vacc[high_i], growth_rate[high_i], "rx")
    ax.plot(vacc, 0*np.ones_like(vacc))    
    
    save_plot(fig, ax, name, DIR_PATH+subdir)


def plot_prob_dis(terms, Is, fs=(12, 8), name="", subdir=""):
    fig, ax = plt.subplots(figsize=fs)
    ax.bar(Is, terms)
    
    save_plot(fig, ax, name, DIR_PATH+subdir)


def plot_pop_struct(N, fs=(12, 8), name="", subdir=""):
    fig, ax = plt.subplots(figsize=fs)
    plt.imshow(N, norm=LogNorm(1, np.max(N)))
    
    save_plot(fig, ax, name, DIR_PATH+subdir)