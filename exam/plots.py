import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from deterministic_SIR import get_Nt, get_asymptotes2

plt.rcParams['mathtext.fontset'] = 'cm'
font = {'family' : 'serif', 
        'size': 20}
plt.rc('font', **font)
plt.rc('lines', lw=2)


labels = ["$S$", "$I$", "$R$"]
colors = [cm.plasma(0.2), cm.plasma(0.5), cm.plasma(0.8)]

labels2 = ["$S$", "$E$", "$I$", "$I_a$", "$R$"]
colors2 = [cm.viridis(i/(len(labels2)-1)) for i in range(len(labels2))]


def plotSIR(x, T, dt, args, fs=(12, 8)):
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

    plt.show()


def plotSIRs(result0, result, fs=(12, 8)):
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

    plt.show()



def plotSEIIaRs(result0, result, fs=(12, 8)):
    fig, ax = plt.subplots(figsize=fs)

    xs, T, dt, args = result
    Nt = get_Nt(T, dt)
    t = np.linspace(0, T, Nt)

    for x in xs:
        N = np.sum(x[0])
        x = x / N # Normalize

        l = []
        for i in range(x.shape[1]):
            l.append(ax.plot(t, x[:, i], color=colors2[i], alpha=0.3))
    

    x, T, dt, args = result0
    Nt = get_Nt(T, dt)
    t = np.linspace(0, T, Nt)
    N = np.sum(x[0])
    x = x / N # Normalize
    
    for i in range(x.shape[1]):
        ax.plot(t, x[:, i], "--", label=labels[i], color=colors[i])

    ax.legend([*labels2, *labels])


    plt.show()



def plotIs(result, fs=(12, 8)):
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

    plt.show()


def plotI(x, T, dt, args, fs=(12, 8)):
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

    plt.show()


def plot_maxI(max_I, betas, high_i, fs=(12, 8)):
    fig, ax = plt.subplots(figsize=fs)
    ax.plot(betas, max_I, "k.-")
    ax.plot(betas[high_i], max_I[high_i], "rx")
    ax.plot(betas, 0.2*np.ones_like(betas))

    plt.show()


def plot_vacc(growth_rate, vacc, high_i, fs=(12, 8)):
    fig, ax = plt.subplots(figsize=fs)
    ax.plot(vacc, growth_rate, "k.-")
    ax.plot(vacc[high_i], growth_rate[high_i], "rx")
    ax.plot(vacc, 0*np.ones_like(vacc))
    
    plt.show()


def plot_prob_dis(terms, Is, fs=(12, 8)):
    fig, ax = plt.subplots(figsize=fs)
    ax.bar(Is, terms)
    plt.show()