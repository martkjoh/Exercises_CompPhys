import numpy as np
from numpy.random import binomial as B, multinomial as M
from deterministic_SIR import integrate, get_Nt
from stochastic_SIR import stoch_step
from tqdm import trange


# x = (S, E, I, I_a, R)
def SEIIaR(x, dt, *args):
    beta, rs, ra, fs, fa, tE, tI = args
    N = np.sum(x)

    PSE = 1 - np.exp(-dt*beta*(rs*x[2]+ra*x[3])/N)
    PEI = fs*(1 - np.exp(-dt/tE))
    PEIa = fa*(1 - np.exp(-dt/tE))
    PIR = 1 - np.exp(-dt/tI)

    DSE = B(x[0], PSE)
    DEI, DEIa, DEE = M(x[1], (PEI, PEIa, 1-PEI-PEIa))
    DIR = B(x[2], PIR)
    DIaR = B(x[3], PIR)
    return np.array([-DSE, DSE - DEI - DEIa, DEI - DIR, DEIa - DIaR, DIR + DIaR])
    

def get_test_SEIIAR():
    #       beta, rs, ra, fs, fa, tE, tI
    args = (0.55, 1, 0.1, 0.6, 0.4, 3, 7)
    N = 100_000
    E = 25
    x0 = np.array([N-E, E, 0, 0, 0], dtype=int)
    T = 180; dt = 0.1
    xs = []
    for i in trange(10):
        xs.append(integrate(SEIIaR, x0, T, dt, args, step=stoch_step, inf=False))

    return xs, T, dt, args


def stay_home():
    runs = 100
    rss = np.linspace(1, 0, runs)
    samples = 100
    N = 100_000
    E = 25
    x0 = np.array([N-E, E, 0, 0, 0], dtype=int)
    T = 20; dt = 0.1
    Nt = get_Nt(T, dt)
    xs = np.zeros((runs, Nt, len(x0)), dtype=type(x0))
    for i in trange(runs):
        rs = rss[i]
        args = (0.55, rs, 0.1, 0.6, 0.4, 3, 7)
        for _ in range(samples):
            xs[i] += integrate(SEIIaR, x0, T, dt, args, step=stoch_step, inf=False)
    
    xs *= 1/samples
    xs = np.array(xs, dtype=np.float64)
    # the growth seems to start approx after 5 days
    n = int(5/dt) # place to start measuring from
    logE = np.log(xs[:, n:, 1])
    av_growth = (logE[:, -1] - logE[:, 0]) / (T - dt*n)
    return xs, T, dt, args, rss, av_growth


if __name__=="__main__":
    get_test_SEIIAR()