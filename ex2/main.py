import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from numpy import cos, sin, exp, pi


dim = 3

eijk = np.zeros((3, 3, 3))
eijk[0, 1, 2] = eijk[1, 2, 0] = eijk[2, 0, 1] = 1
eijk[0, 2, 1] = eijk[2, 1, 0] = eijk[1, 0, 2] = -1


def heun_step(f, y, h, n, args):
    y[n+1] = y[n] + h * f(y[n], *args)
    y[n+1] = y[n] + (h / 2) * (f(y[n], *args) + f(y[n+1], *args))


def NN(S):
    NNsum = np.zeros_like(S)
    NNsum = np.roll(S, 1, 0) + np.roll(S, -1, 0)
    return NNsum


def get_H(S, J, dz, B):
    """ returns the field """
    NNsum = NN(S)
    aniso = np.zeros_like(S)
    aniso[:, 2] = S[:, 2]
    return J * NNsum + 2*dz*aniso + B 


def LLG(S, J, dz, B, a):
    H = get_H(S, J, dz, B)
    dtS = J*np.einsum("...ac, ...c-> ...a", np.einsum("abc, ...b -> ...ac", eijk, S), H)
    if a:
        sum1 = np.einsum("...b, ...b -> ...", S, S)
        sum2 = np.einsum("...b, ...b -> ...", S, H)
        sum1 = np.einsum("j, ji -> ji", sum1, H)
        sum2 = np.einsum("j, ji -> ji", sum2, S)
        dtS += a * (sum1 - sum2)
    return dtS


def integrate(f, S, h, step, args):
    for n in range(len(S)-1):
        step(f, S, h, n, args)


def get_S(n):
    theta = np.random.random(n) * pi
    phi = np.random.random(n) * 2 * pi
    return np.array([cos(phi)*sin(theta), sin(phi)*sin(theta), cos(theta)]).T

def get_S1(n):
    theta = np.zeros(n)
    theta[0] = 0.1
    phi = np.zeros(n)
    return np.array([cos(phi)*sin(theta), sin(phi)*sin(theta), cos(theta)]).T

def get_S2(n):
    theta = np.ones(n) * 0.1
    phi = np.zeros(n)
    return np.array([cos(phi)*sin(theta), sin(phi)*sin(theta), cos(theta)]).T


def one_spin():

    T = 100
    N = 1
    S = np.empty([T, N, dim])

    theta = 0.1
    phi = 0
    np.array([cos(phi)*sin(theta), sin(phi)*sin(theta), cos(theta)])
    S[0] = get_S(1)

    h = 0.1
    a = 0.05
    dz = 0
    J = 1
    B = np.array([0, 0, 1])
    args = (J, dz, B, a)

    integrate(LLG, S, h, heun_step, args)
    t = np.linspace(0, T*h, T)


    coo = ["x", "y", "z"]
    fig, ax = plt.subplots()
    for i in range(dim):
        ax.plot(t, S[:, 0,  i], label="$S_"+coo[i]+"$")
    ax.plot(t, S[0, 0, 0]*exp(-t*a), "--")

    ax.legend()
    plt.show()


def spin_chain():
    T = 1_000
    N = 10
    S = np.empty([T, N, dim])

    S[0] = get_S(N)

    h = 0.1
    a = 0.05
    dz = 0.1
    J = 1
    B = np.array([0, 0, 0])
    args = (J, dz, B, a)

    integrate(LLG, S, h, heun_step, args)
    t = np.linspace(0, T*h, T)


    spins = [str(i) for i in range(N)]
    fig, ax = plt.subplots(dim)
    for i in range(N):
        for j in range(dim):
            ax[j].plot(t, S[:, i, j], 
            label="$S_"+spins[i]+"$",
            color=cm.viridis(i/N))

    plt.show()

def magnon():
    T = 1_000
    N = 10
    S = np.empty([T, N, dim])

    S[0] = get_S1(N)

    h = 0.01
    a = 0.05
    dz = 0.1
    J = 0
    B = np.array([0, 0,0])
    args = (J, dz, B, a)

    integrate(LLG, S, h, heun_step, args)
    t = np.linspace(0, T*h, T)


    spins = [str(i) for i in range(N)]
    fig, ax = plt.subplots()
    for i in range(N):
        ax.plot(t, S[:, i, 0], 
        label="$S_"+spins[i]+"$",
        color=cm.viridis(i/N))

    ax.legend()
    plt.show()

# one_spin()
spin_chain()
# magnon()