import numpy as np
from utillities import *
from plot import *

 
def get_args1(const_K, Nt=10_000):
    Nz = 10_000
    t0_d = 1 # Number of days to run sim for
    t0 = 60*60*24 * t0_d
    L = 100

    dz = L/Nz
    dt = t0/Nt
    a = dt / (2 * dz**2)
    
    kw = 0
    K0 = 0.00001
    if const_K:K = K0*np.ones(Nz)
    else: K = K0*(1 + np.sin(np.linspace(0, 10, Nz)/2))
    Ceq = 0
    return Ceq, K, Nt, Nz, a, dz, dt, kw


def test1(const_K):
    args = get_args1(const_K)
    Ceq, K, Nt, Nz, a, dz, dt, kw = args
    name = "test1"
    if not const_K: name += "_varK"

    C0 = np.ones(Nz)
    C = simulate(C0, args)

    plot_C(C, args, name)
    print("var = {}".format(np.max(C) - np.min(C)))


def test2():
    args = get_args1(False)
    Ceq, K, Nt, Nz, a, dz, dt, kw = args
    name = "test2"

    z = np.linspace(0, Nz * dz, Nz)
    C0 = np.exp(-(z - dz*Nz/2)**2/(5)**2)
    C = simulate(C0, args)
    plot_C(C, args, name+"_C")
    plot_M(C, args, name+"_M")


def test3():
    args = get_args1(True)
    Ceq, K, Nt, Nz, a, dz, dt, kw = args

    z = np.linspace(0, Nz * dz, Nz)
    C0 = np.exp(-(z - dz*Nz/2)**2/(2 * 1/20)**2)
    C = simulate(C0, args)
    plot_C(C, args)
    plot_var(C, args)



def conv_test():
    Nts = 10**(np.linspace(1, 3, 10))
    Nts = np.concatenate([Nts, [4_000,]]) # refrence value
    Cs = []
    Nz = 200
    for Nt in Nts:
        args = get_args1(10, int(Nt))
        Ceq, K, Nt, Nz, a, dz, dt, kw = args
        z = np.linspace(0, Nz * dz, Nz)
        C0 = np.exp(-(z - dz*Nz/2)**2/(2 * 1/20)**2)
        Cs.append(simulate_until(C0, args))
        print(Nt)

    # plot_Cs(Cs, args)
    fac = 10 / (60 * 60 * 24)
    plot_conv(Cs, fac/Nts, 2, args)


def test4(const_K):
    Nz = 10_000
    Nt = 10_000
    t0 = 50
    dz = 1/Nz
    dt = t0/Nt
    a = dt / (2 * dz**2)
    
    kw = 0.05
    K0 = 30100
    if const_K: K = K0 * np.ones(Nz)
    else: K = K0*(2 + np.sin(np.linspace(0, 10, Nz)))
    Ceq = 0

    args = Ceq, K, Nt, Nz, a, dz, dt, kw

    z = np.linspace(0, Nz * dz, Nz)
    C0 = np.ones(Nz)
    C = simulate(C0, args)
    plot_C(C, args)
    plot_M_decay(C, args)


def test5(const_K):
    Nz = 10_000
    Nt = 10_000
    t0 = 4
    dz = 1/Nz
    dt = t0/Nt
    a = dt / (2*dz**2)

    kw = 2
    K0 = 1
    if const_K: K = K0 * np.ones(Nz)
    else: K = K0*(2 + np.sin(np.linspace(0, 10, Nz)))
    Ceq = 0.5

    args = Ceq, K, Nt, Nz, a, dz, dt, kw

    z = np.linspace(0, Nz * dz, Nz)
    C0 = np.ones(Nz)
    C = simulate(C0, args)
    plot_C(C, args)
    plot_minmax(C, args)


# test1(True)
# test1(False)
test2()
# test4(True)
# test5(False)
# conv_test()