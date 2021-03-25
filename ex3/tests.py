import numpy as np
from utillities import *
from plot import *


def get_args1(const_K):
    N = 10_000
    T = 10_000
    t0 = 0.2
    dz = 1/N
    dt = t0/T
    a = dt / (2 * dz**2)
    
    kw = 0
    K0 = 1
    if const_K:K = K0*np.ones(N)
    else: K = K0*(2 + np.sin(np.linspace(0, 10, N)))
    Ceq = 0
    return Ceq, K, T, N, a, dz, dt, kw


def test1():
    args = get_args1(False)
    Ceq, K, T, N, a, dz, dt, kw = args

    C0 = np.ones(N)
    C = simulate(C0, args)
    plot_C(C, args)


def test23(const_K):
    args = get_args1(const_K)
    Ceq, K, T, N, a, dz, dt, kw = args

    z = np.linspace(0, N * dz, N)
    C0 = np.exp(-(z - dz*N/2)**2/(2 * 1/20)**2)
    # C0 = z*(z - 1)
    C = simulate(C0, args)
    plot_C(C, args)
    plot_M(C, args)
    plot_var(C, args)


def test4(const_K):
    N = 10_000
    T = 10_000
    t0 = 50
    dz = 1/N
    dt = t0/T
    a = dt / (2 * dz**2)
    
    kw = 0.05
    K0 = 30100
    if const_K: K = K0 * np.ones(N)
    else: K = K0*(2 + np.sin(np.linspace(0, 10, N)))
    Ceq = 0

    args = Ceq, K, T, N, a, dz, dt, kw

    z = np.linspace(0, N * dz, N)
    C0 = np.ones(N)
    C = simulate(C0, args)
    plot_C(C, args)
    plot_M_decay(C, args)



def test5(const_K):
    N = 10_000
    T = 10_000
    t0 = 4
    dz = 1/N
    dt = t0/T
    a = dt / (2*dz**2)

    kw = 2
    K0 = 1
    if const_K: K = K0 * np.ones(N)
    else: K = K0*(2 + np.sin(np.linspace(0, 10, N)))
    Ceq = 0.5

    args = Ceq, K, T, N, a, dz, dt, kw

    z = np.linspace(0, N * dz, N)
    C0 = np.ones(N)
    C = simulate(C0, args)
    plot_C(C, args)
    plot_minmax(C, args)


# test0()
# test1()
# test23(True)
# test4(True)
test5(False)
