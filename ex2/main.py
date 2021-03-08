import numpy as np
from numpy import cos, sin, exp, pi
from tqdm import trange
from plotting import *

"""
PHYSICS
"""

dim = 3

eijk = np.zeros((3, 3, 3))
eijk[0, 1, 2] = eijk[1, 2, 0] = eijk[2, 0, 1] = 1
eijk[0, 2, 1] = eijk[2, 1, 0] = eijk[1, 0, 2] = -1


def heun_step(f, y, h, n, args):
    fn = f(y[n], *args)
    y[n+1] = y[n] + h * fn
    y[n+1] = y[n] + (h / 2) * (fn + f(y[n+1], *args))


def euler_step(f, y, h, n, args):
    y[n+1] = y[n] + h * f(y[n], *args)

def RK4(f, y, h, n, args):
    k1 = f(y[n], *args) * h
    k2 = f(y[n] + k1 / 2, *args) * h
    k3 = f(y[n] + k2 / 2, *args) * h
    k4 = f(y[n] + k3, *args) * h
    delta = 1 / 6 * (k1 + 2*k2 + 2*k3 + k4)
    y[n+1] = y[n] + delta


def get_H(S, J, dz, B):
    """ returns the field """
    NNsum = np.roll(S, 1, 0) + np.roll(S, -1, 0)
    Sz = np.zeros_like(S)
    Sz[:, 2] = S[:, 2]
    return J * NNsum + 2*dz*Sz + B 


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


def integrate(f, S, h, tmax, step, args):
    for n in trange(len(S)-1):
        h = min(h, tmax - n*h)
        step(f, S, h, n, args)

"""
INITIALIZATION
"""

def get_S(n):
    theta = np.random.random(n) * pi
    phi = np.random.random(n) * 2 * pi
    return np.array([cos(phi)*sin(theta), sin(phi)*sin(theta), cos(theta)]).T

def get_S1(n):
    theta = np.zeros(n)
    theta[0] = 0.5
    phi = np.zeros(n)
    return np.array([cos(phi)*sin(theta), sin(phi)*sin(theta), cos(theta)]).T


def get_S2(n):
    theta = np.ones(n) * 0.5
    phi = np.zeros(n)
    return np.array([cos(phi)*sin(theta), sin(phi)*sin(theta), cos(theta)]).T


"""
EXERCISES
"""


def ex211():
    T, N, h = 1000, 1, 0.01
    tmax = h*T
    S = np.empty([T, N, dim])
    S[0] = get_S1(N)
    args = (0, 0, [0, 0, 1], 0) # (J, dz, B, a)

    integrate(LLG, S, h, tmax, heun_step, args)

    plot_single(S, h, args, "single")


def ex212():
    steps = [euler_step, heun_step]
    names = ["e", "h"]
    pows = [1, 2]
    time = 5
    N = 1
    args = (0, 0, [0, 0, 1], 0) # (J, dz, B, a)

    n = 40
    hs = 10**(-np.linspace(0, 4, n))
    Sx = np.empty((2, n))
    Ts = np.empty((2, n))
    S0 = get_S1(N)[0, 0]

    for i, step in enumerate(steps):
        for j, h in enumerate(hs):
            T = int(time/h)
            Ts[i, j] = T
            S = np.empty([T, N, dim])
            S[0] = get_S1(N)
            integrate(LLG, S, h, time, step, args)
            Sx[i, j] = S[-1, 0, 0]

    plot_err_afh(Sx, hs, Ts, S0, args, pows, names, "err")



def ex213():
    T, N, h = 5_000, 1, 0.01
    S = np.empty([T, N, dim])
    S[0] = get_S1(N)
    args = (1, 0, [0, 0, 1], 0.05) # (J, dz, B, a)

    integrate(LLG, S, h, heun_step, args)

    plot_decay(S, h, args, "decay")



def ex221a():
    T, N, h = 5_000, 10, 0.01
    S = np.empty([T, N, dim])
    S[0] = get_S(N)

    args = (1, 0.1, [0, 0, 1], 0.05) # (J, dz, B, a)

    integrate(LLG, S, h, heun_step, args)
    plot_coords(S, h, "ground_state", args)
    anim_spins(S, 10)


def magnon():
    T, N, h = 100_000, 100, 0.01
    S = np.empty([T, N, dim])
    S[0] = get_S1(N)

    args = (-1, 0.1, [0, 0, 0], 0.01) # (J, dz, B, a)

    integrate(LLG, S, h, heun_step, args)
    plot_coords(S, h)
    anim_spins(S, 10)



# ex211()
ex212()
# ex213()
# ex221a()
# magnon()