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
        print(x.shape)
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
    
    

    plt.show()


def plot_two_towns(result, fs=(12, 8)):
    xs, T, dt, args = result
    N_cities = xs.shape[1]
    Nt = get_Nt(T, dt)
    t = np.linspace(0, T, Nt)
    fig, ax = plt.subplots(1, 2, figsize=fs)
    print(xs.shape)
    for n in range(2):
        x = xs[:, :, n]
        N = np.sum(x[0])
        x = x / N # Normalize

        for i in range(x.shape[1]):
            ax[n].plot(t, x[:, i], color=colors2[i], alpha=1)


    plt.show()



def plot_many_towns(result, fs=(12, 8), shape=(3, 3)):
    xs, T, dt, args = result
    N_cities = xs.shape[1]
    Nt = get_Nt(T, dt)
    t = np.linspace(0, T, Nt)
    fig, ax = plt.subplots(*shape, figsize=fs)

    for j in range(shape[0]):
        for k in range(shape[1]):
            n = j*shape[0] + k
            x = xs[:, n]
            N = np.sum(x[0])
            x = x / N # Normalize

            for i in range(x.shape[1]):
                ax[j, k].plot(t, x[:, i], color=colors2[i], alpha=0.3)


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


def plotEs(result, frac=10, fs=(12, 8)):
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

    plt.show()


def plotEsafrs(result, frac, fs=(12, 8)):
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

    plt.show()


def plotEav(result, frac=10, fs=(12, 8)):
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