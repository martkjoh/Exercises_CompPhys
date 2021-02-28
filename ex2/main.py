import numpy as np
from numpy import cos, sin, exp, pi

from plotting import *

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
    dtS = np.einsum("...ac, ...c-> ...a", np.einsum("abc, ...b -> ...ac", eijk, S), H)
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
    theta[0] = 1
    phi = np.zeros(n)
    return np.array([cos(phi)*sin(theta), sin(phi)*sin(theta), cos(theta)]).T


def get_S2(n):
    theta = np.ones(n) * 0.1
    phi = np.zeros(n)
    return np.array([cos(phi)*sin(theta), sin(phi)*sin(theta), cos(theta)]).T


def one_spin():
    T, N, h = 1000, 1, 0.01
    S = np.empty([T, N, dim])
    S[0] = get_S1(1)
    args = (1, 0.1, [0, 0, 1], 0.05) # (J, dz, B, a)

    integrate(LLG, S, h, heun_step, args)

    plot_decay(S, h, args)


def spin_chain():
    T, N, h = 5_000, 50, 0.1
    S = np.empty([T, N, dim])
    S[0] = get_S(N)

    args = (-1, 0.1, [0, 0, 1], 0.05) # (J, dz, B, a)

    integrate(LLG, S, h, heun_step, args)
    plot_coords(S, h)
    anim_spins(S)


def magnon():
    T, N, h = 10_000, 50, 0.01
    S = np.empty([T, N, dim])
    S[0] = get_S1(N)

    args = (-1, 0.1, [0, 0, 1], 0.0) # (J, dz, B, a)

    integrate(LLG, S, h, heun_step, args)
    plot_coords(S, h)
    anim_spins(S, 5)



# one_spin()
# spin_chain()
magnon()