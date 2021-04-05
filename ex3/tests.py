import numpy as np
from utillities import *
from plot import *

 
def get_args1(const_K, Nt=1_000, Nz=1_001, t0_d=1):
    def get_K(z, L, K0):
        K1 = K0/10
        a = 0.5
        z0 = L/2
        return K1 + (K0 - K1) / (1 + np.exp(-a*(z - z0)))

    t0 = 60*60*24 * t0_d
    L = 100

    dz = L/(Nz - 1)
    dt = t0/(Nt - 1)
    a = dt / (2 * dz**2)
    
    kw = 0
    K0 = 1e-3
    z = np.linspace(0, L, Nz)
    if const_K: K = K0*np.ones(Nz)
    else: K = get_K(z, L, K0)
    Ceq = np.zeros(Nt)
    return Ceq, K, Nt, Nz, a, dz, dt, kw, L, t0


def conv_test_t():
    Nts = 10**(np.linspace(2, 4, 10))
    Nts = np.concatenate([Nts, [50_000,]]) # refrence value
    Cs = []
    Nz = 200
    for Nt in Nts:
        args = get_args1(True, int(Nt))
        Ceq, K, Nt, Nz, a, dz, dt, kw, L, t0 = args
        z = np.linspace(0, L, Nz)
        C0 = np.exp(-(z - L/2)**2/(2 * 20)**2)
        Cs.append(simulate_until(C0, args))
        print("Nt={}".format(Nt))

    plot_conv_t(Cs, Nts, 2, args, "conv_test")


def conv_test_z():
    Nzs = np.array([21, 51, 101, 201, 501, 1_001, 2_001, 10_001])
    Cs = []
    for Nz in Nzs:
        args = get_args1(True, Nz=int(Nz), Nt=10_000)
        Ceq, K, Nt, Nz, a, dz, dt, kw, L, t0 = args
        z, dz0 = np.linspace(0, L, Nz, retstep=True)
        C0 = np.exp(-(z - L/2)**2/(2 * 20)**2)
        Cs.append(simulate_until(C0, args))
        print("Nz={}".format(Nz))

    plot_conv_z(Cs, Nzs, 2, args, "conv_test_z")


def test1(const_K):
    args = get_args1(const_K, Nt=10_000)
    Ceq, K, Nt, Nz, a, dz, dt, kw, L, t0 = args
    name = "test1"
    if not const_K: name += "_varK"

    C0 = np.ones(Nz)
    C = simulate(C0, args)

    plot_C(C, args, name)
    print("var = {}".format(np.max(C) - np.min(C)))


def test2():
    args = get_args1(False, t0_d=10, Nt=10_001)
    Ceq, K, Nt, Nz, a, dz, dt, kw, L, t0 = args
    name = "test2"

    z = np.linspace(0, Nz * dz, Nz)
    C0 = np.exp(-(z - L/2)**2/10**2)/10 \
        + np.exp(-(z - L*3/5)**2/5**2)/5
    C = simulate(C0, args)
    plot_C(C, args, name+"_C")
    plot_M(C, args, name+"_M")
    plot_Cs([C[0], C[-1]], args)
    plot_Cs([K,], args)


def test3():
    args = get_args1(True, t0_d=20)
    Ceq, K, Nt, Nz, a, dz, dt, kw, L, t0 = args

    z = np.linspace(0, Nz * dz, Nz)
    C0 = np.exp(-(z - L/2)**2/(2 * 5)**2)
    C = simulate(C0, args)
    plot_C(C, args, "test3")
    plot_var(C, args, "test3_var")


def test4(const_K):
    Nz = 10_000
    Nt = 10_000
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

    z = np.linspace(0, Nz * dz, Nz)
    C0 = np.ones(Nz)
    C = simulate(C0, args)
    plot_C(C, args, "test4")
    plot_M_decay(C, args, "test4_decay")


def test5(const_K):
    Nz = 1_000
    Nt = 10_000
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
    Ceq = 0.5*np.ones(Nt)

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
# test2()
# test3()
# test4(True)
test5(True)
test5(False)
