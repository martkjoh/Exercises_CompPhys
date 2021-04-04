import numpy as np
from utillities import *
from plot import *

 
def get_args1(const_K, Nt=1_000, Nz = 1_000, t0_d=1):
    Nz = (Nz//2)*2 + 1 # Make Nz odd
    t0 = 60*60*24 * t0_d
    L = 100

    dz = L/(Nz - 1)
    dt = t0/(Nt - 1)
    a = dt / (2 * dz**2)
    
    kw = 0
    K0 = 1e-3
    if const_K:K = K0*np.ones(Nz)
    else: K = K0*(1 + np.sin(np.linspace(0, 10, Nz)/2))
    Ceq = np.zeros(Nt)
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
    args = get_args1(True)
    Ceq, K, Nt, Nz, a, dz, dt, kw = args
    name = "test2"

    z = np.linspace(0, Nz * dz, Nz)
    C0 = np.exp(-(z - dz*Nz/2)**2/10**2)/10 \
        + np.exp(-(z - dz*Nz*3/5)**2/5**2)/5
    C = simulate(C0, args)
    plot_C(C, args, name+"_C")
    plot_M(C, args, name+"_M")
    plot_Cs([C[0],C[-1]], args)


def conv_test_t():
    Nts = 10**(np.linspace(2, 4, 10))
    Nts = np.concatenate([Nts, [50_000,]]) # refrence value
    Cs = []
    Nz = 200
    for Nt in Nts:
        args = get_args1(True, int(Nt))
        Ceq, K, Nt, Nz, a, dz, dt, kw = args
        z = np.linspace(0, Nz * dz, Nz)
        C0 = np.exp(-(z - dz*Nz/2)**2/(2 * 1/20)**2)
        Cs.append(simulate_until(C0, args))
        print(Nt)

    plot_conv_t(Cs, Nts, 2, args, "conv_test")


def conv_test_z():
    Nzs = ((10**(np.linspace(1, 3, 10)))//2)*2 + 1
    print(Nzs)
    Nzs = np.concatenate([Nzs, [10_000,]]) # refrence value
    Cs = []
    Nt = 10
    # for Nz in Nzs:
    #     args = get_args1(True, Nz=int(Nz), Nt=Nt)
    #     Ceq, K, Nt, Nz, a, dz, dt, kw = args
    #     z = np.linspace(0, Nz * dz, Nz)
    #     C0 = np.exp(-(z - dz*Nz/2)**2/(2 * 1/20)**2)
    #     Cs.append(simulate_until(C0, args))
    #     print(Nt)

    # plot_conv_t(Cs, Nzs, 2, args, "conv_test")
 

def test3():
    args = get_args1(True)
    Ceq, K, Nt, Nz, a, dz, dt, kw = args

    z = np.linspace(0, Nz * dz, Nz)
    C0 = np.exp(-(z - dz*Nz/2)**2/(2 * 1/20)**2)
    C = simulate(C0, args)
    plot_C(C, args)
    plot_var(C, args)


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
# test2()
# conv_test_t()
conv_test_z()
# test4(True)
# test5(False)