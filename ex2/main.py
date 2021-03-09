import numpy as np
from numpy import cos, sin, exp, pi
from tqdm import trange
from plotting import *
from scipy.signal import spectrogram


"""
PHYSICS
"""

dim = 3

""" Levi-Civita symbol """
eijk = np.zeros((3, 3, 3))
eijk[0, 1, 2] = eijk[1, 2, 0] = eijk[2, 0, 1] = 1
eijk[0, 2, 1] = eijk[2, 1, 0] = eijk[1, 0, 2] = -1


def heun_step(f, y, h, n, args):
    fn = f(y[n], *args)
    y[n+1] = y[n] + h * fn
    y[n+1] = y[n] + (h / 2) * (fn + f(y[n+1], *args))


def euler_step(f, y, h, n, args):
    y[n+1] = y[n] + h * f(y[n], *args)


def get_H(S, J, dz, B):
    """ returns the effective field, given spin conf. """
    NNsum = np.roll(S, 1, 0) + np.roll(S, -1, 0)
    Sz = np.zeros_like(S)
    Sz[:, 2] = S[:, 2]
    return J * NNsum + 2*dz*Sz + B


def LLG(S, J, dz, B, a):
    """ returns the time derivative of S, as given by the LLG eqs. """
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
    for n in trange(len(S)-1):
        step(f, S, h, n, args)


"""
INITIALIZATION
"""


def get_S(n):
    """ randomly distributed spins """
    theta = np.random.random(n) * pi
    phi = np.random.random(n) * 2 * pi
    return np.array([cos(phi)*sin(theta), sin(phi)*sin(theta), cos(theta)]).T

def get_S1(n, offset=0.5):
    """ one spin tilted """
    theta = np.zeros(n)
    theta[0] = offset
    phi = np.zeros(n)
    return np.array([cos(phi)*sin(theta), sin(phi)*sin(theta), cos(theta)]).T


def get_S2(n):
    """ one spin tilted, af """
    theta = np.zeros(n)
    theta[::2] = np.ones(n//2) * np.pi
    theta[0] = np.pi - 1
    phi = np.zeros(n)
    return np.array([cos(phi)*sin(theta), sin(phi)*sin(theta), cos(theta)]).T


"""
EXERCISES
"""

def ex211():
    T, N, h = 1000, 1, 0.01
    S = np.empty([T, N, dim])
    S[0] = get_S1(N)
    args = (0, 0, [0, 0, 1], 0) # (J, dz, B, a)

    integrate(LLG, S, h, heun_step, args)

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
            integrate(LLG, S, h, step, args)
            Sx[i, j] = S[-1, 0, 0]

    plot_err_afh(Sx, hs, Ts, S0, args, pows, names, "err")


def ex213():
    T, N, h = 10_000, 1, 0.01
    S = np.empty([T, N, dim])
    S[0] = get_S1(N, offset=0.2)
    alphas = [0.1, 0.05, 0.01]
    for a in alphas:
        args = (1, 0, [0, 0, 1], a) # (J, dz, B, a)
        integrate(LLG, S, h, heun_step, args)
        plot_decay(S, h, args, "decay_a={}".format(a))


def ex221():
    T, N, h = 30_000, 10, 0.01
    S = np.empty([T, N, dim])
    S[0] = get_S(N)
    args = (1, 0.1, [0, 0, 0], 0.05) # (J, dz, B, a)
    integrate(LLG, S, h, heun_step, args)
    plot_zs(S, h, "ground_state_f", args)
    plot_spins(S[-1], "ground_state_f3D")

    T, N, h = 5_000, 10, 0.01
    S = np.empty([T, N, dim])
    S[0] = get_S(N)
    args = (-1, 0.1, [0, 0, 0], 0.05) # (J, dz, B, a)
    integrate(LLG, S, h, heun_step, args)
    plot_zs(S, h, "ground_state_af", args)
    plot_spins(S[-1], "ground_state_af3D")


def ex2221():
    T, N, h = 7_000, 10, 0.01
    S = np.empty([T, N, dim])
    S[0] = get_S1(N)

    args = (0, 0.1, [0, 0, 0], 0.0) # (J, dz, B, a)

    integrate(LLG, S, h, heun_step, args)
    plot_coords(S, h, "2221", args)


def ex2222():
    T, N, h = 10_000, 10, 0.01
    S = np.empty([T, N, dim])
    S[0] = get_S1(N)

    args = (1, 0.1, [0, 0, 0], 0.0) # (J, dz, B, a)

    integrate(LLG, S, h, heun_step, args)
    plot_coords(S, h, "2222", args)


def ex2224():
    T, N, h = 40_000, 10, 0.01
    S = np.empty([T, N, dim])
    S[0] = get_S1(N)

    args = (1, 0.1, [0, 0, 0], 0.01) # (J, dz, 5B, a)

    integrate(LLG, S, h, heun_step, args)
    plot_coords(S, h, "2224", args, coords=[0])
    plot_fit_to_sum(S, h, args, "2224fit")



def ex2225():
    T, N, h = 40_000, 10, 0.01
    S = np.empty([T, N, dim])
    S[0] = get_S1(N)

    args = (-1, 0.1, [0, 0, 0], 0.01) # (J, dz, 5B, a)

    integrate(LLG, S, h, heun_step, args)
    plot_coords(S, h, "2225", args)


def ex22252():
    T, N, h = 40_000, 10, 0.01
    S = np.empty([T, N, dim])
    S[0] = get_S2(N)

    args = (-1, 0.1, [0, 0, 0], 0.01) # (J, dz, 5B, a)

    integrate(LLG, S, h, heun_step, args)
    plot_coords(S, h, "22252", args)


def ex2226():
    T, N, h = 40_000, 10, 0.01
    S = np.empty([T, N, dim])
    S[0] = get_S1(N)

    args = (1, 0.1, [0, 0, 0], 0.01) # (J, dz, 5B, a)

    integrate(LLG, S, h, heun_step, args)
    Mz = np.einsum("tn -> t", S[:, :, 2]) / len(S[0])
    plot_mag(Mz, h, "mag", args)


# ex211()
# ex212()
ex213()
# ex221()
# ex2221()
# ex2222()
# ex2224()
# ex22252()
# ex2226()
