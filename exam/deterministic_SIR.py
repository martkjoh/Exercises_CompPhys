import numpy as np
from time import time
from tqdm import trange
import sys
sys.setrecursionlimit(10_000)
from scipy.optimize import root

# TODO: flytt alt som brukes andre steder

# x[i] = (S(t_i), I(t_i), R(t_i))
def SIR(x, beta, tau):
    a = beta * x[1] * x[0]
    b = x[1]/tau
    return np.array([-a, a - b, b])


def RK4step(f, x, i, dt, args):
    k1 = f(x, *args) * dt
    k2 = f(x + 1 / 2 * k1, *args) * dt
    k3 = f(x + 1 / 2 * k2, *args) * dt
    k4 = f(x + k3, *args)  * dt
    return 1 / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


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


def get_asymptotes2(args):
    R0 = args[0] * args[1]
    fS = lambda S: np.exp(-R0*(1 - S)) - S
    fR = lambda R: 1 - np.exp(-R0*R) - R
    return root(fS, 0.5)["x"], root(fR, 0.5)["x"]


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


def get_testSIR():
    eps = 1e-4
    x0 = np.array([1-eps, eps, 0])
    T = 180; dt = 0.01
    args = (0.25, 10) # beta, tau
    x = integrate(SIR, x0, T, dt, args)
    return x, T, dt, args


def flatten_the_curve():
    eps = 1e-4
    x0 = np.array([1-eps, eps, 0])
    T = 180; dt = 0.01
    betas = np.linspace(0.15, 0.25, 10)
    max_I = []
    max_day = []
    for b in betas:
        args = (b, 10) # beta, tau
        x = integrate(SIR, x0, T, dt, args)
        max_I.append(np.max(x[:, 1]))
        max_day.append(np.argmax(x[:, 1]))

    # The index of the highest beta above 0.2
    high_i = np.arange(0, len(betas))[np.less(max_I, 0.2)][-1]
    print("Lates day to reach top = {}".format(np.max(max_day)))
    print("Highest beta giving I below 0.2: {}".format(betas[high_i]))
    print("Highest I below 0.2: {}".format(max_I[high_i]))
    print("Reach at index {} of {}".format(high_i, len(betas)))

    return max_I, betas, high_i


def vaccination():
    eps = 1e-4
    T = 0.01; dt = 0.01
    vacc = np.linspace(0, 1, 10)
    args = (0.25, 10) # beta, tau
    xs = []
    for v in vacc:
        x0 = np.array([1-eps-v, eps, v])
        xs.append(integrate(SIR, x0, T, dt, args, inf=False))

    growth_rate = [np.log(x[1, 1]/x[0, 1]) for x in xs]
    # The index of the highest v with positive growth rate
    high_i = np.arange(0, len(vacc))[np.greater(growth_rate, 0)][-1]
    print("highest v with positive growth rate: {}".format(vacc[high_i]))
    print("Corr growth rate: {}".format(growth_rate[high_i]))
    print("Reach at index {} of {}".format(high_i, len(vacc)))
    return growth_rate, vacc, high_i



if __name__=="__main__":
    # flatten_the_curve()    
    vaccination()
