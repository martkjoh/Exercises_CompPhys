import numpy as np
from numpy.random import default_rng
from deterministic_SIR import integrate, get_Nt
from tqdm import tgrange

rng = default_rng()
B = rng.binomial
M = rng.multinomial


def SEIIaR_commute2(x, dt, day, *args):
    beta, rs, ra, fs, fa, tE, tI = args
    if not day:
        x0 = np.sum(x, axis=1)
        N = np.sum(x0, axis=0)
        x0 = np.ones_like(x) * x0[:, np.newaxis, :]
        N = np.ones_like(x[0]) * N[np.newaxis, :]

    else:
        x0 = np.sum(x, axis=2)
        N = np.sum(x0, axis=0)
        x0 = np.ones_like(x) * x0[:, :, np.newaxis]
        N = np.ones_like(x[0]) * N[:, np.newaxis]

    v = -dt*beta*(rs*x0[2]+ra*x0[3])/N
    PSE = 1 - np.exp(v)
    PEI = fs*(1 - np.exp(-dt/tE))
    PEIa = fa*(1 - np.exp(-dt/tE))
    PIR = 1 - np.exp(-dt/tI)

    DSE = B(x[0], PSE)
    DEI, DEIa, _ = np.moveaxis(M(x[1], (PEI, PEIa, 1-PEI-PEIa)), -1, 0)
    DIR = B(x[2], PIR)
    DIaR = B(x[3], PIR)

    return np.array([-DSE, DSE - DEI - DEIa, DEI - DIR, DEIa - DIaR, DIR + DIaR])


def stoch_commute_step(f, x, i, dt, args):
    t = i*dt
    time = t - int(t)
    day = time <= 0.5
    dx = f(x, dt, day, *args)
    return dx


def get_pop_structure():
    return np.loadtxt("population_structure.csv", delimiter=",")


def get_test_SEIIaR_commute():
    args = (0.55, 1, 0.1, 0.6, 0.4, 3, 7)
    N = np.array([
        [100_000, 0],
        [0, 1]
    ], dtype=int)
    E = np.array([
        [25, 0],
        [0, 0]
    ], dtype=int)
    Oh = np.zeros_like(N)
    x0 = np.array([N-E, E, Oh, Oh, Oh], dtype=int)
    T = 180; dt = 0.1
    xs = []
    for i in range(10):
        xs.append(integrate(
            SEIIaR_commute2, x0, T, dt, args, step=stoch_commute_step)[:, :, 0, 0]
            )
    return xs, T, dt, args


def get_two_towns():
    args = (0.55, 1, 0.1, 0.6, 0.4, 3, 7)
    N = np.array([
        [9000, 1000],
        [200, 99800]
    ], dtype=int)
    E = np.array([
        [25, 0],
        [0, 0]
    ], dtype=int)
    Oh = np.zeros_like(N)
    # x[time, var, city_i, city_j]
    x0 = np.array([N-E, E, Oh, Oh, Oh], dtype=int)
    T = 180; dt = 0.1
    Nt = get_Nt(T, dt)
    xs = np.zeros((Nt, 5, 2, 2))
    for i in range(10):
        xs += integrate(SEIIaR_commute2, x0, T, dt, args, step=stoch_commute_step)
    xs = np.sum(xs, axis=2)/10
    return xs, T, dt, args


def get_nine_towns():
    args = (0.55, 1, 0.1, 0.6, 0.4, 3, 7)
    N = np.array([
    [198600, 100, 100, 100, 100, 1000, 0, 0, 0, 0],
    [500, 9500, 0, 0, 0, 0, 0, 0, 0, 0],
    [500, 0, 9500, 0, 0, 0, 0, 0, 0, 0],
    [500, 0, 0, 9500, 0, 0, 0, 0, 0, 0],
    [500, 0, 0, 0, 9500, 0, 0, 0, 0, 0],
    [1000, 0, 0, 0, 0, 498200, 200, 200, 200, 200],
    [0, 0, 0, 0, 0, 1000, 0, 19000, 0, 0],
    [0, 0, 0, 0, 0, 1000, 0, 0, 19000, 0],
    [0, 0, 0, 0, 0, 1000, 0, 0, 19000, 0],
    [0, 0, 0, 0, 0, 1000, 0, 0, 0, 19000]]
    )
    E = np.zeros_like(N)
    E[1, 1] = 25
    Oh = np.zeros_like(N)
    x0 = np.array([N-E, E, Oh, Oh, Oh], dtype=int)
    T = 180; dt = 0.1
    Nt = get_Nt(T, dt)
    x = np.zeros((Nt, 5, *N.shape))
    for i in range(10):
        x += integrate(SEIIaR_commute2, x0, T, dt, args, step=stoch_commute_step)
    x = np.sum(x, axis=2)/10
    return x, T, dt, args
    

def get_Norway():
    args = (0.55, 1, 0.1, 0.6, 0.4, 3, 7)
    N = get_pop_structure()
    E = np.zeros_like(N)
    E[0, 0] = 25
    Oh = np.zeros_like(N)
    x0 = np.array([N-E, E, Oh, Oh, Oh], dtype=int)
    T = 180; dt = 0.1
    save = 101
    x = np.zeros((save, 5, *N.shape))
    for i in range(1):
        x += integrate(SEIIaR_commute2, x0, T, dt, args, save=save, step=stoch_commute_step)
    x = np.sum(x, axis=2)/1
    return x, T, dt, args



if __name__=="__main__":
    # get_test_SEIIaR_commute()
    # get_two_towns()
    # get_pop_structure()
    get_Norway()
