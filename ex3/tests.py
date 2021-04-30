import numpy as np
from utilities import *
from plot import *

 
def get_args1(const_K, Nt=1_001, Nz=1_001, t0_d=10):
    t0 = 60*60*24 * t0_d
    L = 100
    dz = L/(Nz - 1)
    dt = t0/(Nt - 1)
    a = dt / (2 * dz**2)
    
    kw = 0
    K0 = 1e-3
    Ka = 2e-2
    za = 7
    Kb = 5e-2
    zb = 10

    Ceq = np.zeros(Nt)
    z = np.linspace(0, L, Nz)

    if const_K: K = K0*np.ones(Nz)
    else: K = K0 + Ka*z/za * np.exp(-z/za) + Kb*(L-z)/zb * np.exp(-(L-z)/zb)

    return Ceq, K, Nt, Nz, a, dz, dt, kw, L, t0


def conv_test_t():
    Nts = (10**(np.linspace(1, 4, 20))).astype(int)
    Nts = np.concatenate([Nts, [50_000,]]) # refrence value
    Cs = []
    Nz = 201
    for Nt in Nts:
        args = get_args1(True, Nz=Nz, Nt=Nt)
        Ceq, K, Nt, Nz, a, dz, dt, kw, L, t0 = args
        z = np.linspace(0, L, Nz)
        C0 = np.exp(-(z - L/2)**2/(2 * 20)**2)
        Cs.append(simulate(C0, args, save=2)[-1])
        print("Nt={}".format(Nt))

    plot_conv_t(Cs, Nts, 2, args, "conv_test_t")


def conv_test_z():
    Nzs = get_Nzs(10, 2**10*10 + 1) # 10k ish

    Cs = []
    for Nz in Nzs:
        args = get_args1(True, Nz=Nz, Nt=10_001)
        Ceq, K, Nt, Nz, a, dz, dt, kw, L, t0 = args
        z, dz0 = np.linspace(0, L, Nz, retstep=True)
        C0 = np.exp(-(z - L/2)**2/(2 * 20)**2)/20
        Cs.append(simulate(C0, args, save=2)[-1])
        print("Nz={}".format(Nz))

    plot_conv_z(Cs, Nzs, 2, args, "conv_test_z")


def test1(const_K):
    args = get_args1(const_K)
    Ceq, K, Nt, Nz, a, dz, dt, kw, L, t0 = args
    name = "test1"
    if not const_K: name += "_varK"

    C0 = np.ones(Nz)
    C = simulate(C0, args)

    plot_C(C, args, name, fs=(7, 5))
    print("var = {}".format(np.max(C) - np.min(C)))


def test2(const_K):
    args = get_args1(const_K, t0_d=10, Nz=100_001, Nt=10_001)
    Ceq, K, Nt, Nz, a, dz, dt, kw, L, t0 = args
    name = "test2"
    if not const_K: name += "_varK"

    z = np.linspace(0, Nz * dz, Nz)
    C0 = np.exp(-(z - L/2)**2/(2 * 20)**2)/20
    C = simulate(C0, args, save=101)
    plot_C(C, args, name+"_C")
    plot_dM(C, args, name+"_M")
    plot_K(args, name)


def test3():
    args = get_args1(True, t0_d=20)
    Ceq, K, Nt, Nz, a, dz, dt, kw, L, t0 = args

    z = np.linspace(0, Nz * dz, Nz)
    C0 = np.exp(-(z - L/2)**2/(2 * 5)**2)/5
    C = simulate(C0, args)
    plot_C(C, args, "test3")
    plot_var(C, args, "test3_var")


def test4(const_K):
    Nz = 10_000
    Nt = 10_001
    t0_d = 1
    t0 = 60*60*24 * t0_d
    L = 100

    dz = L/(Nz - 1)
    dt = t0/(Nt - 1)
    a = dt / (2 * dz**2)
    
    kw = 0.005
    K0 = 310
    if const_K: K = K0 * np.ones(Nz)
    else: K = K0*(2 + np.sin(np.linspace(0, 10, Nz)))
    Ceq = np.zeros(Nt)

    args = Ceq, K, Nt, Nz, a, dz, dt, kw, L, t0
    name = "test4"
    if not const_K: name += "_varK"

    z = np.linspace(0, Nz * dz, Nz)
    C0 = np.ones(Nz)
    C = simulate(C0, args)
    plot_C(C, args, name)
    plot_M_decay(C, args, name+"_decay")


def test5(const_K):
    Nz = 1_001
    Nt = 10_001
    t0_d = 20
    t0 = 60*60*24 * t0_d
    L = 100

    dz = L/(Nz - 1)
    dt = t0/(Nt - 1)
    a = dt / (2 * dz**2)

    kw = 1e-3
    K0 = 2e-2
    if const_K: K = K0 * np.ones(Nz)
    else: K = K0*(2 + np.sin(np.linspace(0, 10, Nz)))
    H = 5060; ppco2 = 415e-6
    Ceq = np.ones(Nt) * H * ppco2

    args = Ceq, K, Nt, Nz, a, dz, dt, kw, L, t0
    name = "test5"
    if not const_K: name += "_varK"

    C0 = np.zeros(Nz)
    C = simulate(C0, args)
    plot_C(C, args, name)
    plot_minmax(C, args, name+"_minmax")


# conv_test_t()
# conv_test_z()
# test1(True)
# test1(False)
# test2(True)
# test2(False)
# test3()
# test4(True)
# test4(False)
test5(True)
test5(False)
