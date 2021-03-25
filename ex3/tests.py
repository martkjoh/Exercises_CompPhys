import numpy as np
from utillities import *
from plot import *


def test0():
    args = get_args1(False)
    t0 = 100
    dt = 0.1
    T = int(t0/dt)

    N = 10_000
    dz = 100/N
    a = 10*dt / (2*dz**2)

    oh = np.zeros(N)
    args = (oh, oh, T, N, a, dz, dt, 0)

    z = np.linspace(0, N * dz, N)
    C0 = np.exp(-(z - dz*N/2)**2/(2 * 1/20)**2)
    C = simulate0(C0, args)
    plot_C(C, args)
    plot_M(C, args)


def get_args1(const_K):
    N = 1_000
    T = 50_000
    t0 = 0.1
    dz = 1/N
    dt = t0*1/T
    a = dt / dz**2 / 2
    
    kw = 0
    K0 = 10
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
    N = 1000
    T = 10_000
    t0 = 0.1
    dz = 1/N
    dt = t0/T
    a = dt / dz**2 / 2

    kw = 0.02
    K0 = 3100
    if const_K: K = K0 * np.ones(N)
    else: K = K0*(2 + np.sin(np.linspace(0, 10, N)))
    Ceq = 0

    args = Ceq, K, T, N, a, dz, dt, kw

    z = np.linspace(0, N * dz, N)
    C0 = np.ones(N)
    C = simulate(C0, args)
    plot_C(C, args)
    plot_M_decay(C, args)


def get_args3(const_K):
    N = 1000
    T = 10_000
    t0 = 10
    dz = 1/N
    dt = t0/T
    a = dt / dz**2 / 2

    kw = 2
    K0 = 1
    if const_K: K = K0 * np.ones(N)
    else: K = K0*(2 + np.sin(np.linspace(0, 10, N)))
    Ceq = -0.5
    return Ceq, K, T, N, a, dz, dt, kw


def test5():
    args = get_args3(True)
    Ceq, K, T, N, a, dz, dt, kw = args

    z = np.linspace(0, N * dz, N)
    C0 = np.ones(N)
    C = simulate(C0, args)
    plot_C(C, args)
    plot_minmax(C, args)


# test0()
# test1()
test23(True)
# test4(True)
# test5()