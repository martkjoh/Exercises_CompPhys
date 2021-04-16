import numpy as np
from utilities import *
from plot import *


H = 5060
kw = 6.97e-5
ppco2 = 415e-6


##########
# Prob 2 #
##########


def get_args2(t0_d, Nt, Nz):
    t0 = 60*60*24 * t0_d
    L = 100
    dz = L/(Nz - 1)
    dt = t0/(Nt - 1)
    a = dt / (2 * dz**2)

    z = np.linspace(0, L, Nz)
    K0 = 1e-3
    Ka = 2e-2
    za = 7
    Kb = 5e-2
    zb = 10
    K = K0 + Ka*z/za * np.exp(-z/za) + Kb*(L-z)/zb * np.exp(-(L-z)/zb)
    Ceq = H * ppco2 * np.ones(Nt)

    return Ceq, K, Nt, Nz, a, dz, dt, kw, L, t0


def prob2_conv_test_t():
    Nts = (10**(np.linspace(0.4, 4.5, 20))).astype(int)
    Nts = np.concatenate([Nts, [200_000,]]) # refrence value
    Cs = []
    Nz = 1_001
    for Nt in Nts:
        print("Nt={}".format(Nt))
        args = get_args2(10, Nt, Nz)
        Ceq, K, Nt, Nz, a, dz, dt, kw, L, t0 = args
        z = np.linspace(0, L, Nz)
        C0 = np.zeros(Nz)
        Cs.append(simulate(C0, args, save=2)[-1])

    plot_conv_t(Cs, Nts, 2, args, "prob2_conv_test_t")


def prob2_conv_test_z():
    Nzs = get_Nzs(8, 2**10*10*100 + 1) # 1m ish
    Cs = []
    Nt = 1_001
    for Nz in Nzs:
        print("Nz={}".format(Nz))
        args = get_args2(10, Nt, Nz)
        Ceq, K, Nt, Nz, a, dz, dt, kw, L, t0 = args
        z, dz0 = np.linspace(0, L, Nz, retstep=True)
        C0 = np.zeros(Nz)
        Cs.append(simulate(C0, args, save=2)[-1])

    plot_conv_z(Cs, Nzs, 2, args, "prob2_conv_test_z")


def prob2():
    args = get_args2(180, 10_001, 10_001)
    Ceq, K, Nt, Nz, a, dz, dt, kw, L, t0 = args

    C0 = np.zeros(Nz)
    C = simulate(C0, args)
    plot_C(C, args, "prob2", fs=(12, 6))
    plot_minmax(C, args, "prob2_minmax")
    indxs = [0, 10, 50, 100, 250, -1]
    plot_Ci(C, indxs, args, "prob2_i")
    plot_K(args, "prob2_K")


##########
# Prob 3 #
##########

def get_args3(t0_y, Nt, Nz):
    t0 = 60*60*24 * 365 * t0_y
    L = 4000
    dz = L/(Nz - 1)
    dt = t0/(Nt - 1)
    a = dt / (2 * dz**2)

    K0 = 1e-3
    K1 = 1e-2
    b = 0.5
    z0 = 100
    z = np.linspace(0, Nz*dz, Nz)
    K = K1 + (K0 - K1) / (1 + np.exp(-b*(z - z0)))

    t = np.linspace(0, t0, Nt)
    Ceq = H * (ppco2 + 2.3e-6/(60*60*24*360) * t)

    return Ceq, K, Nt, Nz, a, dz, dt, kw, L, t0

def prob3_conv_test_t():
    Nts = (10**(np.linspace(1, 3.5, 20))).astype(int)
    Nts = np.concatenate([Nts, [100_000,]]) # refrence value
    Cs = []
    Nz = 201
    for Nt in Nts:
        print("Nt={}".format(Nt))
        args = get_args3(1, Nt, Nz)
        Ceq, K, Nt, Nz, a, dz, dt, kw, L, t0 = args
        z = np.linspace(0, L, Nz)
        C0 = Ceq[0] * np.ones(Nz)
        Cs.append(simulate(C0, args, save=2)[-1])

    plot_conv_t(Cs, Nts, 2, args, "prob3_conv_test_t")


def prob3_conv_test_z():
    Nzs = get_Nzs(8, 2**10*10*100 + 1) # 100k ish
    Cs = []
    Nt = 1_001
    for Nz in Nzs:
        print("Nz={}".format(Nz))
        args = get_args3(1, Nt, Nz)
        Ceq, K, Nt, Nz, a, dz, dt, kw, L, t0 = args
        z, dz0 = np.linspace(0, L, Nz, retstep=True)
        C0 = Ceq[0] * np.ones(Nz)
        Cs.append(simulate(C0, args, save=2)[-1])

    plot_conv_z(Cs, Nzs, 2, args, "prob3_conv_test_z")


def prob3():
    args = get_args3(10, 10_001, 10_001)
    Ceq, K, Nt, Nz, a, dz, dt, kw, L, t0 = args
    C0 = Ceq[0] * np.ones(Nz)
    N = 501
    C = simulate(C0, args, save=N)
    plot_C(C, args, "prob3", fs=(12, 6))
    plot_K(args, "prob3_K")

    indxs = [0, int(N/4), int(N/2), -1]
    plot_Ci(C, indxs, args, "prob3_i")
    plot_M(C, args, "prob3_M")



# prob2_conv_test_t()
# prob2_conv_test_z()
# prob2()

# prob3_conv_test_t()
# prob3_conv_test_z()
# prob3()
