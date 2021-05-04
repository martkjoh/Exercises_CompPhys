import numpy as np
from numpy.random import default_rng
from tqdm import trange
import sys
sys.setrecursionlimit(10_000)

rng = default_rng()
B = rng.binomial
M = rng.multinomial



"""
Functions of the form f = dy/dt
"""

# x[i] = (S(t_i), I(t_i), R(t_i))
def SIR(x, beta, tau):
    a = beta * x[1] * x[0]
    b = x[1]/tau
    return np.array([-a, a - b, b])


def SIR_stoch(x, dt, beta, tau):
    N = np.sum(x)
    PSI = 1 - np.exp(-dt*beta*x[1]/N)
    PIR = 1 - np.exp(-dt/tau)
    a, b = B(x[0], PSI), B(x[1], PIR)
    return np.array([-a, a - b, b])


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


def SEIIaR_commute(x, dt, day, *args):
    beta, rs, ra, fs, fa, tE, tI = args
    if day:
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


"""
Function that gives a time step, given f = dy/dt
"""

def RK4step(f, x, i, dt, args):
    k1 = f(x, *args) * dt
    k2 = f(x + 1 / 2 * k1, *args) * dt
    k3 = f(x + 1 / 2 * k2, *args) * dt
    k4 = f(x + k3, *args)  * dt
    return 1 / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def stoch_step(f, x, i, dt, args):
    return f(x, dt, *args)


def stoch_commute_step(f, x, i, dt, args):
    t = i*dt
    time = t - int(t)
    day = time < 0.5
    dx = f(x, dt, day, *args)
    return dx


"""
Integrators: simulates system given by f=dy/dt
Needs f, and step, which is the scheme to be used
"""

def integrate(f, x0, T, dt, args, save=None, step=RK4step, inf=True):
    Nt = get_Nt(T, dt)
    if inf: print("Integrates {} steps until time {}".format(Nt-1, T))
    if save is None: save = Nt
    x = np.empty((save, *x0.shape), dtype=x0.dtype)
    x[0] = x0
    assert (Nt-1)%(save-1)==0
    skip = (Nt-1)//(save-1)
    if inf: r = trange(save-1)
    else: r = range(save-1)
    for i in r:
        xi = x[i]
        for j in range(skip):
            xi += step(f, xi, i, dt, args)
        x[i+1] = xi
    return x


def integrate_untill(f, x, T, dt, args, cond, step=RK4step, inf=False):
    Nt = get_Nt(T, dt) # maximum amount of steps
    if inf: print("Integrates {} steps until time {}".format(Nt-1, T))
    i = 0
    while i<Nt and not cond(x):
        x += step(f, x, i, dt, args)
        i+=1
    return x, i, cond(x)
    

"""
Misc
"""


def get_Nt(T, dt):
    assert np.isclose(int(T/dt)*dt, T)
    return int(T/dt)+1


def fpi(x0, f, tol=1e-8):
    x1 = f(x0)
    if np.abs(x0-x1)<tol:
        return x1
    else:
        return fpi(x1, f)


def get_asymptotes(args):
    R0 = args[0] * args[1]
    # assert np.abs(R0-1)<1e-3 # wont converge
    fS = lambda S: np.exp(-R0*(1 - S))
    fR = lambda R: 1 - np.exp(-R0*R)
    return fpi(0.5, fS), fpi(0.5, fR)


def get_pop_structure(lockdown=False):
    N = np.loadtxt("population_structure.csv", delimiter=",")
    if lockdown:
        N_cities = np.sum(N, axis=1)
        off_diag = np.eye(*N.shape)==0
        N[off_diag] = N[off_diag]//10
        new_N_cities = np.sum(N, axis=1)
        delta = N_cities-new_N_cities
        N += np.diag(delta)
        N_cities_lockdown = np.sum(N, axis=1)
        assert np.sum(N_cities_lockdown != N_cities)==0
    return N
