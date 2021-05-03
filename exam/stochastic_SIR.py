import numpy as np
from numpy.random import binomial as B
from deterministic_SIR import integrate

def SIR_stoch(x, dt, beta, tau):
    N = np.sum(x)
    PSI = 1 - np.exp(-dt*beta*x[1]/N)
    PIR = 1 - np.exp(-dt/tau)
    a, b = B(x[0], PSI), B(x[1], PIR)
    return np.array([-a, a - b, b])


def stoch_step(f, x, i, dt, args):
    x[i+1] = x[i] + f(x[i], dt, *args)


def get_test_stoch():
    N = 100_000
    I = 10
    x0 = np.array([N-I, I, 0], dtype=int)
    T = 180; dt = 0.1
    args = (0.25, 10) # beta, tau
    xs = []
    for i in range(100):
        xs.append(integrate(SIR_stoch, x0, T, dt, args, step=stoch_step))

    return xs, T, dt, args




if __name__=="__main__":
    print(type(B(100, 0.1)))
