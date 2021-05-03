import numpy as np
from numpy.random import default_rng
from deterministic_SIR import integrate, get_Nt
from tqdm import tgrange

rng = default_rng()
B = rng.binomial
M = rng.multinomial


def set_SEIIaR_delta(x, x0, dt, N, day, args, deltas):
    beta, rs, ra, fs, fa, tE, tI = args
    DSE, DEI, DEIa, DIR, DIaR = deltas
    if day:
        indx = lambda i: (slice(None), i)
    else:
        indx = lambda i: (i, slice(None))

    # For each city
    for i, xi in enumerate(x0):
        v = np.array(-dt*beta*(rs*xi[2]+ra*xi[3])/N)
        PSE = 1 - np.exp(v)
        PEI = fs*(1 - np.exp(-dt/tE))
        PEIa = fa*(1 - np.exp(-dt/tE))
        PIR = 1 - np.exp(-dt/tI)

        # Please, do not ask why. I don't know
        S = x[(*indx(i), 0)].astype(int)
        E = x[(*indx(i), 1)].astype(int)
        I = x[(*indx(i), 2)].astype(int)
        Ia = x[(*indx(i), 3)].astype(int)
        DSE[indx(i)] = B(S, PSE)
        DEI[indx(i)], DEIa[indx(i)], _ = M(E, (PEI, PEIa, 1-PEI-PEIa)).T #Why?
        DIR[indx(i)] = B(I, PIR)
        DIaR[indx(i)] = B(Ia, PIR)



# x = (S, E, I, I_a, R)
def SEIIaR_commute(x, dt, day, *args):

    # The total number in each city is determined by a sum of everyone in the city
    # Day or night determins which axis to sum over (commuter, lives there)
    axis = 0 if day else 1
    
    N_cities = len(x[0])

    # Sum up all people in each city
    x0 = np.sum(x, axis=axis, dtype=np.float64)
    N = np.sum(x0)
    DSE = np.empty((N_cities, N_cities))
    DEI = np.empty((N_cities, N_cities))
    DEIa = np.empty((N_cities, N_cities))
    DIR = np.empty((N_cities, N_cities))
    DIaR = np.empty((N_cities, N_cities))
    deltas = [DSE, DEI, DEIa, DIR, DIaR]

    set_SEIIaR_delta(x, x0, dt, N, day, args, deltas)

    dx = np.array([-DSE, DSE - DEI - DEIa, DEI - DIR, DEIa - DIaR, DIR + DIaR])
    return np.moveaxis(dx, 0, 2)



def stoch_commute_step(f, x, i, dt, args):
    t = i*dt
    time = t - int(t)
    day = time <= 0.5
    dx = f(x, dt, day, *args)
    return dx


def get_test_SEIIaR_commute():
    args = (0.55, 1, 0.1, 0.6, 0.4, 3, 7)
    N = np.array([
        [100_000, 0],
        [0, 0]
    ], dtype=int)
    E = np.array([
        [25, 0],
        [0, 0]
    ], dtype=int)
    Oh = np.zeros_like(N)
    x0 = np.array([N-E, E, Oh, Oh, Oh], dtype=int)
    x0 = np.moveaxis(x0, 0, 2) # move index with SEIIaR to the back
    T = 180; dt = 0.1
    xs = []
    for i in range(10):
        xs.append(integrate(
            SEIIaR_commute, x0, T, dt, args, step=stoch_commute_step)[:, 0, 0]
            )
    print(np.sum(xs[0], axis=1))
    return xs, T, dt, args


def get_two_towns():
    args = (0.55, 1, 0.1, 0.6, 0.4, 3, 7)
    N = np.array([
        [9000, 1000],
        [200, 99800]
    ], dtype=int)
    E = np.array([
        [0, 0],
        [0, 25]
    ], dtype=int)
    Oh = np.zeros_like(N)
    x0 = np.array([N-E, E, Oh, Oh, Oh], dtype=int)
    x0 = np.moveaxis(x0, 0, 2) # move index with SEIIaR to the back
    T = 180; dt = 0.1
    x = integrate(SEIIaR_commute, x0, T, dt, args, step=stoch_commute_step)
    print(x.shape)
    xs = np.sum(x, axis=2)
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
    E[5, 5] = 25
    Oh = np.zeros_like(N)
    x0 = np.array([N-E, E, Oh, Oh, Oh], dtype=int)
    x0 = np.moveaxis(x0, 0, 2) # move index with SEIIaR to the back
    T = 500; dt = 0.1
    x = integrate(SEIIaR_commute, x0, T, dt, args, step=stoch_commute_step)
    print(x.shape)
    xs = np.sum(x, axis=2)
    return xs, T, dt, args


if __name__=="__main__":
    # get_test_SEIIaR_commute()
    get_two_towns()