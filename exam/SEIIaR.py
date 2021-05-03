import numpy as np
from numpy.random import binomial as B, multinomial as M
from deterministic_SIR import integrate
from stochastic_SIR import stoch_step

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
    assert DEI + DEIa + DEE == x[1]
    DIR = B(x[2], PIR)
    DIaR = B(x[3], PIR)
    return np.array([-DSE, DSE - DEI - DEIa, DEI - DIR, DEIa - DIaR, DIR + DIaR])


def stoch_step(f, x, dt, args):
    return f(x, dt, *args)
    

def get_test_SEIIAR():
    #       beta, rs, ra, fs, fa, tE, tI
    args = (0.55, 1, 0.1, 0.6, 0.4, 3, 7)
    N = 100_000
    E = 25
    x0 = np.array([N-E, E, 0, 0, 0], dtype=int)
    T = 180; dt = 0.1
    xs = []
    for i in range(10):
        xs.append(integrate(SEIIaR, x0, T, dt, args, step=stoch_step))

    return xs, T, dt, args

