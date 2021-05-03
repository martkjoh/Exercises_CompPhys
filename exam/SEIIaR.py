import numpy as np
from numpy.random import multinomial as M

beta    = 0.55
rs      = 1
ra      = 0.1
fs      = 0.6
fa      = 0.4
tE      = 3
tI      = 7


# x = (S, E, I, I_a, R)
def SEIIaR_step(x, dt):
    N = np.sum(x)
    PSE = 1 - np.exp(-dt*beta*(rs*x[2]+ra*x[3])/N)
    PEI = fs*(1 - np.exp(-dt/tE))
    PEIa = fa*(1 - np.exp(-dt/tE))
    PIR = 1 - np.exp(-dt/tI)
    # a, b = B(x[0], PSI), B(x[1], PIR)
    # return np.array([-a, a - b, b])


def stoch_step(f, x, dt, args):
    return f(x, dt, *args)

