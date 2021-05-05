import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from utilities import get_Nt, get_asymptotes
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


"""
Determenistic SIR
"""

def plotSIR(x, T, dt, args, fs=(8, 6), name="", subdir=""):
    fig, ax = plt.subplots(figsize=fs)
    Nt = get_Nt(T, dt)
    t = np.linspace(0, T, Nt)
    N = np.sum(x[0])
    x = x / N # Normalize
    
    for i in range(x.shape[1]):
        ax.plot(t, x[:, i], label=labels[i], color=colors[i])
    S_inf, R_inf = get_asymptotes(args)
    ax.plot(t, np.ones_like(t)*S_inf, "--", label="$S(\infty)$", color=colors[0])
    ax.plot(t, np.ones_like(t)*R_inf, "--", label="$R(\infty)$", color=colors[2])
    ax.legend()
    ax.set_title(
        "$\Delta t = {:.2f}$".format(dt)
        + "$,\,\\beta={}$".format(args[0])
        + "$,\,\\tau={}$".format(args[1])
    )
    ax.set_xlabel("$t/[\mathrm{ days }]$")

    save_plot(fig, ax, name, DIR_PATH+subdir)


def plotI(x, T, dt, args, fs=(8, 6), name="", subdir=""):
    fig, ax = plt.subplots(figsize=fs)
    Nt = get_Nt(T, dt)
    Nt0 = (Nt-1)//2 + 1
    T0 = T*((Nt0-1)/(Nt-1))
    t, dt0 = np.linspace(0, T0, Nt0, retstep=True)
    assert np.isclose(dt0, dt)
    N = np.sum(x[0])
    x = x / N # Normalize

    ax.semilogy(t, x[:Nt0, 1], label=labels[1], color=colors[1], lw=6, alpha=0.6)
    a = args[0] - 1 / args[1]
    ax.semilogy(t, x[0, 1]*np.exp(a*t), "--k", label="$\exp([\\beta -1/\\tau]t)$")
    ax.legend()
    ax.set_title(
        "$\Delta t = {:.2f}$".format(dt)
        + "$,\,\\beta={}$".format(args[0])
        + "$,\,\\tau={}$".format(args[1])
    )

    ax.set_xlabel("$t/[\mathrm{ days }]$")

    save_plot(fig, ax, name, DIR_PATH+subdir)


def plot_conv_det(xs, dts, args, T, steps, name="", subdir=""):
    fig, ax = plt.subplots(figsize=(12, 5))
    N = np.sum(xs[0][0][0])
    exps = [1, 2, 4]
    lines = ["--", "-.", "-"]
    names = ["Euler", "midpoint", "RK4"]

    R_ref = xs[1][-1][-1, 2]
    for i in range(len(steps)):
        xs2 = xs[i] # not ideal
        R0 = [x[-1, 2]/N for x in xs2[:-1]]

        c = np.abs(R0[4] - R_ref)/R_ref
        c_dts_pow = [c*dt**(exps[i]) for dt in dts[:-1]]
        ax.loglog(
            dts[:-1], c_dts_pow, "k", ls=lines[i], label="$\propto \Delta t^{}$".format(exps[i])
            )

        ax.loglog(
            dts[:-1], np.abs(R0-R_ref)/R_ref, "X", 
            label=names[i], color=colors[i], ms=12
            )

    ax.legend()
    ax.set_xlabel("$\Delta t $")
    ax.set_ylabel("$\Delta R$")
    save_plot(fig, ax, name, DIR_PATH+subdir)


def plot_flattening(results, fs=(8, 6), name="", subdir=""):
    xs, betas, high_i, max_I, max_day, T, dt, args = results
    fig, ax = plt.subplots(figsize=fs)
    Nt = get_Nt(T, dt)
    t = np.linspace(0, T, Nt)
    for i in range(len(xs)):
        x = xs[i]
        N = np.sum(x[0])
        x = x / N # Normalize    
        ax.plot(t, x[:, 1], color=colors[1], alpha=0.3)

    ax.plot(t, xs[high_i][:, 1], "k--", label="$\\beta = {:.3f}$".format(betas[high_i]))
    ax.plot(t, 0.2*np.ones_like(t), "k", ls="dashdot", label="$0.2$".format(betas[high_i]))
    ax.legend()
    ax.set_title(
        "$\Delta t = {:.2f}$".format(dt)
        + "$,\,\\tau={}$".format(args[1])
    )
    ax.set_xlabel("$t/[\mathrm{ days }]$")

    save_plot(fig, ax, name, DIR_PATH+subdir)


def plot_maxI(results, fs=(8, 6), name="", subdir=""):
    xs, betas, high_i, max_I, max_day, T, dt, args = results
    fig, ax = plt.subplots(figsize=fs)
    ax.plot(betas, max_I, "k.-")
    ax.plot(betas, 0.2*np.ones_like(betas), label="$0.2$")
    ax.plot(
        betas[high_i], max_I[high_i], "ro", 
        label="$\\beta = {:.3f}$".format(betas[high_i]), ms=10
        )
    ax.legend()
    ax.set_xlabel("$\\beta$")
    ax.set_ylabel("$\mathrm{ max }(I)$")
    ax.set_title(
        "Lowest $\\beta$ yieldin $I>0.2$: $\\beta={:.3f}$".format(betas[high_i+1])
        )
    save_plot(fig, ax, name, DIR_PATH+subdir)


def plot_vacc(results, fs=(8, 6), name="", subdir=""):
    xs, growth_rate, vacc, high_i, dt, T, args = results
    fig, ax = plt.subplots(figsize=fs)
    Nt = get_Nt(T, dt)

    t = np.linspace(0, T, Nt)
    for i in range(len(xs)):
        x = xs[i]
        N = np.sum(x[0])
        x = x / N # Normalize
        ax.semilogy(t, x[:, 1], color=colors[1], lw=1)

    ax.plot(
        t, xs[high_i+1][:, 1], "k--", 
        label="$R(0) = {:.3f}$".format(vacc[high_i+1]), lw=1
        )
    t2 = np.linspace(t[0], t[-1], 100)
    ax.plot(t2, xs[0][0, 1]*np.ones_like(t2), "k", ls="dashdot", label="const", lw=1)
    ax.legend()
    ax.set_title(
        "$\Delta t = {:.2f}$".format(dt)
        + "$,\,\\tau={}$".format(args[1])
    )
    ax.set_xlabel("$t/[\mathrm{ days }]$")

    save_plot(fig, ax, name, DIR_PATH+subdir)


def plot_growth(results, fs=(8, 6), name="", subdir=""):
    xs, growth_rate, vacc, high_i, dt, T, args = results
    fig, ax = plt.subplots(figsize=fs)
    ax.plot(vacc, 0*np.ones_like(vacc), label="$\\alpha=0$")
    ax.plot(vacc, growth_rate, "k.-", label="$\\alpha$")
    ax.plot(
        vacc[high_i+1], growth_rate[high_i+1], "ro", ms=10,
        label="$R(0) = {:.3f}$".format(vacc[high_i+1])
        )
    ax.legend()
    ax.set_xlabel("$R(0)$")
    ax.set_ylabel("$\\alpha$")
    ax.set_title(
        "Lowest $R(0)$ yieldin $\\alpha>0$:" + \
            "$R(0)={:.3f}$".format(vacc[high_i])
        )
    
    save_plot(fig, ax, name, DIR_PATH+subdir)


"""
Stohchastic SIR
"""

def plotSIRs(result0, result, fs=(8, 6), name="", subdir=""):
    fig, ax = plt.subplots(figsize=fs)

    xs, T, dt, args = result
    Nt = get_Nt(T, dt)
    t = np.linspace(0, T, Nt)

    for x in xs:
        N = np.sum(x[0])
        x = x / N # Normalize

        l = []
        for i in range(x.shape[1]):
            l.append(ax.plot(t, x[:, i], color=colors[i], alpha=0.3))

    ax.legend(labels)
    x0, T, dt, args = result0
    Nt = get_Nt(T, dt)
    t = np.linspace(0, T, Nt)

    for i in range(x0.shape[1]):
        ax.plot(t, x0[:, i], "k--")

    ax.set_title(
        "$\Delta t = {:.2f}$".format(dt)
        + "$,\,\\beta={}$".format(args[0])
        + "$,\,\\tau={}$".format(args[1])
    )
    ax.set_xlabel("$t/[\mathrm{ days }]$")

    save_plot(fig, ax, name, DIR_PATH+subdir)


def plotIs(result, fs=(8, 6), name="", subdir=""):
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

        l1 = ax.semilogy(t, x[:Nt0, 1], color=colors[1], alpha=0.3)[0]

    a = args[0] - 1 / args[1]
    l2 = ax.semilogy(t, xs[0][0, 1]/N*np.exp(a*t), "--k")[0]
    ax.legend((l1, l2), (labels[1], "$\exp([\\beta -1/\\tau]t)$"))
    ax.set_title(
        "$\Delta t = {:.2f}$".format(dt)
        + "$,\,\\beta={}$".format(args[0])
        + "$,\,\\tau={}$".format(args[1])
    )
    ax.set_xlabel("$t/[\mathrm{ days }]$")
    save_plot(fig, ax, name, DIR_PATH+subdir)


def plot_conv_stoch(xs, dts, args, T, name="", subdir=""):
    fig, ax = plt.subplots(figsize=(12, 5))
    N = np.sum(xs[0][0])
    S, R = get_asymptotes(args)
    R0 = [x[-1, 2]/N for x in xs]

    c = np.abs(R0[1] - R)/R
    c_dts_pow = [c*dt**(1) for dt in dts]
    ax.loglog(dts, c_dts_pow, "k--", label="$\propto \Delta t$")

    ax.loglog(dts, np.abs(R0-R)/R, "X", color=colors[2], label="$\Delta R(\Delta t)$", ms=12)
    ax.loglog(0.1, c*0.1**(1), "o", color=colors[0], label="$\Delta t = 0.1, \, \Delta R = {:.4f} $".format(c * 0.1))
    ax.legend()
    ax.set_xlabel("$\Delta t $")
    ax.set_ylabel("$\Delta R$")
    save_plot(fig, ax, name, DIR_PATH+subdir)


def plot_prob_dis(terminate, Is, fs=(10, 6), name="", subdir=""):
    runs = 10
    samples = 100
    terminate_av = np.array([
        np.mean(terminate[:, i*samples:(i+1)*samples], axis=1).T for i in range(runs)
        ]).T
    mean = np.mean(terminate_av, axis=1)
    std = np.std(terminate_av, axis=1)/np.sqrt(runs)
    print(std)
    fig, ax = plt.subplots(figsize=fs)
    ax.bar(Is, mean)
    ax.errorbar(Is, mean, std, color="k", fmt=".")
    ax.set_xlabel("$I(0)$")
    ax.set_ylabel("prob.")

    
    save_plot(fig, ax, name, DIR_PATH+subdir)


"""
SEIIaR model
"""

def plotSEIIaRs(result0, result, fs=(12, 6), name="", subdir="", alpha=0.8):
    fig, ax = plt.subplots(figsize=fs)

    xs, T, dt, args = result
    Nt = get_Nt(T, dt)
    save = result[0][0].shape[0]
    t = np.linspace(0, T, save)

    for x in xs:
        N = np.sum(x[0])
        x = x / N # Normalize
        for i in range(x.shape[1]):
            ax.plot(t, x[:, i], color=colors2[i], alpha=alpha)
    ax.legend([*labels2])

    x, T, dt, args = result0
    Nt = get_Nt(T, dt)
    t = np.linspace(0, T, Nt)
    N = np.sum(x[0])
    x = x / N # Normalize
    for i in range(x.shape[1]):
        ax.plot(t, x[:, i], "k--")
    ax.set_xlabel("$t/[\mathrm{ days }]$")

    save_plot(fig, ax, name, DIR_PATH+subdir)


def plotEsafrs(result, fs=(12, 6), name="", subdir=""):
    fig, ax = plt.subplots(figsize=fs)

    xs, T, dt, args, rss, av_growth, high_i = result
    N = np.sum(xs[0][0])
    xs = xs / N # Normalize
    Nt = get_Nt(T, dt)
    t = np.linspace(0, T, Nt)
    for i in range(len(xs)):
        ax.semilogy(t, xs[i,:,1], color=cm.viridis(i/len(xs)), alpha=0.7)

    ax.set_xlabel("$t/[\mathrm{ days }]$")
    ax.semilogy(
        t, xs[high_i, : , 1], "k--", ms=7,
        label="$r_s={:.3f}$".format(rss[high_i])
        )
    ax.semilogy(
        t, xs[high_i, -1, 1]*np.ones_like(t), "k", ms=7, 
        ls="dashdot", label="const"
        )
    ax.legend()
    ax.set_title(
        "Lowest $r_s$ yieldin $\\alpha>0$:" + \
            "$r_s={:.3f}$".format(rss[high_i+1])
        )
    
    save_plot(fig, ax, name, DIR_PATH+subdir)


"""
Plot SEIIaR comuter model
"""

def plot_two_towns(result, fs=(16, 6), name="", subdir=""):
    xs, T, dt, args = result
    N_cities = xs.shape[1]
    Nt = get_Nt(T, dt)
    t = np.linspace(0, T, Nt)
    fig, ax = plt.subplots(1, 2, figsize=fs)
    for n in range(2):
        x = xs[:, :, n]
        N = np.sum(x[0])
        x = x / N # Normalize
        I = x[:, 2] + x[:, 3]
        peak = np.argmax(I)

        for i in range(x.shape[1]):
            ax[n].plot(t, x[:, i], color=colors2[i], lw=4)
        ax[n].plot([t[peak], t[peak]], [0, 1], "k--")
        ax[n].set_title(
            "Town {}".format(n+1) + \
            ", Peak I: day {:.0f}".format(t[peak])
            )

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
            I = x[:, 2] + x[:, 3]
            peak = np.argmax(I)
            
            for i in range(x.shape[1]):
                ax[j, k].plot(t, x[:, i], color=colors2[i], alpha=1)
            ax[j, k].plot([t[peak], t[peak]], [0, 1], "k--")
            ax[j, k].set_title(
                "Town {}".format(n+1) + \
                ", Peak I: day {:.0f}".format(t[peak])
                )

    save_plot(fig, ax, name, DIR_PATH+subdir)



def plot_town_i(result, i0, fs=(12, 8), name="", subdir=""):
    fig, ax = plt.subplots(figsize=fs)

    xs, T, dt, args = result
    x = xs[:, :, i0]
    Nt = get_Nt(T, dt)
    save = x.shape[0]
    t = np.linspace(0, T, save)

    N = np.sum(x[0])
    x = x / N # Normalize
    I = x[:, 2] + x[:, 3]
    peak = np.argmax(I)

    for i in range(x.shape[1]):
        ax.plot(t, x[:, i], color=colors2[i])
    ax.plot([t[peak], t[peak]], [0, 1], "k--")
    R_inf = x[-1, 4]
    ax.set_title(
        "Town {}".format(i0+1) + \
        ", Peak I: day {:.0f}".format(t[peak]) +\
        "$,\,R(\infty) = {:.3f}$".format(R_inf)
    )
    ax.legend([*labels2])
    ax.set_xlabel("$t/[\mathrm{ days }]$")

    
    save_plot(fig, ax, name, DIR_PATH+subdir)


def plot_sum_inf(result, fs=(10, 8), name="", subdir=""):
    fig, ax = plt.subplots(figsize=fs)

    xs, T, dt, args = result
    I_tot = xs[:, :, 2] + xs[:, :, 3]
    infected_cities = np.sum(I_tot > 10, axis=2)
    infected_cities_av = np.mean(infected_cities, axis=0)
    std = np.std(infected_cities, axis=0)

    save = xs.shape[1]
    t = np.linspace(0, T, save)
    
    for i in range(len(xs)):
        ax.plot(t, infected_cities[i], color=colors[1], alpha=0.4)

    ax.plot(t, infected_cities_av, "k-", lw=3, label="$\langle I \\rangle$")
    ax.plot(t, infected_cities_av-std, "k--", lw=1)
    ax.plot(t, infected_cities_av+std, "k--", lw=1, label="$\pm \sigma$")

    N_cities = len(xs[0, 0, 0])
    ax.set_ylim(0, N_cities+5)
    max = infected_cities_av>350
    max_i = None
    if np.sum(max)>0:
        max_i = np.argwhere(infected_cities==N_cities)[0, 0]
    
    title = "# days above 350: {}".format(np.sum(max))
    if not max_i is None:
        title += ", day reached: {:.0f}".format(t[max_i])
    else:
        title += ", max: {:.0f}".format(np.max(infected_cities))    

    ax.set_title(title)
    ax.set_xlabel("$t/[\mathrm{ days }]$")
    ax.set_ylabel("# towns with active infection")
    ax.legend()

    save_plot(fig, ax, name, DIR_PATH+subdir)


def plot_pop_struct(N, fs=(8, 8), name="", subdir=""):
    fig, ax = plt.subplots(figsize=fs)
    plt.imshow(N, norm=LogNorm(1, np.max(N)))
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([]) 
    
    save_plot(fig, ax, name, DIR_PATH+subdir)


def plot_towns(pop, fs=(8, 6), name="", subdir=""):
    fig, ax = plt.subplots(figsize=fs)
    pop_sorted = np.sort(pop)[::-1]
    n = np.arange(1, len(pop)+1)
    c = 1/np.sum(pop)

    ax.bar(n, pop_sorted)
    ax.set_yscale("log")
    ax.set_xlabel("population rank")
    ax.set_ylabel("population")

    save_plot(fig, ax, name, DIR_PATH+subdir)
