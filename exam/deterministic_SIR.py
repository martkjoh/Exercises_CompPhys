import numpy as np
from time import time
from tqdm import trange



# x[i] = (S(t_i), I(t_i), R(t_i))
def SIR(x, beta, tau):
    a = beta * x[1] * x[0]
    b = x[1]/tau
    return np.array([-a, a - b, b])


def RK4step(f, x, i, dt, args):
    k1 = f(x[i], *args) * dt
    k2 = f(x[i] + 1 / 2 * k1, *args) * dt
    k3 = f(x[i] + 1 / 2 * k2, *args) * dt
    k4 = f(x[i] + k3, *args)  * dt
    x[i + 1] = x[i] + 1 / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def get_Nt(T, dt):
    assert np.isclose(int(T/dt)*dt, T)
    return int(T/dt)+1


def integrate(f, x0, T, dt, args, step=RK4step):
    Nt = get_Nt(T, dt)
    print("Integrates {} steps until time {}".format(Nt, T))
    x = np.empty((Nt, *x0.shape))
    x[0] = x0
    for i in trange(Nt-1):
        step(f, x, i, dt, args)
    return x


def get_testSIR():
    eps = 1e-3
    x0 = np.array([1-eps, eps, 0])
    T = 40; dt = 0.01
    args = (2, 1) # beta, tau
    x = integrate(SIR, x0, T, dt, args)
    return x, T, dt, args
