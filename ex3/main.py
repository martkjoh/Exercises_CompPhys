import numpy as np
from utillities import *
from plot import *


H = 5060
kw = 5.97e-5
ppco2 = 415e-6


def get_args2(t0_d, Nt, Nz):

    def get_K(z, L):
        K0 = 1e-3
        Ka = 2e-2
        za = 7
        Kb = 5e-2
        zb = 10
        return K0 + Ka * z / za  *np.exp(z / za) + Kb * (L - z)/zb * np.exp(-(L - z)/zb)

    t0 = 60*60*24 * t0_d
    dz = 100/Nz
    dt = t0/Nt
    a = dt / (2*dz**2)

    z = np.linspace(0, Nz*dz, Nz)
    K = get_K(z, Nz*dz)

    Ceq = H * ppco2

    return Ceq, K, Nt, Nz, a, dz, dt, kw


def conv_test1():
    Nts = 10**(np.linspace(1, 4, 10))
    Nts = np.concatenate([Nts, [50_000,]]) # refrence value
    Cs = []
    Nz = 200
    C0 = np.zeros(Nz)
    for Nt in Nts:
        args = get_args2(10, int(Nt), int(Nz))
        Ceq, K, Nt, Nz, a, dz, dt, kw = args
        Cs.append(simulate(C0, args)[-1])
        print(Nt)

    plot_Cs(Cs, args)
    fac = 10 / (60 * 60 * 24)
    plot_conv(Cs, fac/Nts, 1, args)


def conv_test2():
    Nzs = 10**(np.linspace(1, 4, 10))
    Nzs = np.concatenate([Nzs, [10_000,]]) # refrence value
    Cs = []
    Nt = 201
    for Nz in Nzs:
        Nz = int(Nz)
        C0 = np.zeros(Nz)
        args = get_args2(10, Nt, Nz)
        Ceq, K, Nt, Nz, a, dz, dt, kw = args
        Cs.append(simulate_until(C0, args))

    plot_conv2(Cs, Nzs, 1, args)


def prob2():
    args = get_args2(180, 10_000, 10_00)
    Ceq, K, Nt, Nz, a, dz, dt, kw = args

    C0 = np.zeros(Nz)
    C = simulate(C0, args)
    plot_C(C, args)


def prob3():
    def get_K(z, L):
        
        K0 = 1e-4
        K1 = 1e-2
        a = 0.5
        z0 = 100
        return K1 + (K0 - K1) / (1 + np.exp(-a*(z - z0)))

    t0 = 60*60*24 * 360 * 10

    Nz = 10_000
    Nt = 10_000
    dz = 4000/Nz
    dt = t0/Nt
    a = dt / (2*dz**2)

    z = np.linspace(0, Nz*dz, Nz)
    K = get_K(z, Nz*dz)

    t = np.linspace(0, t0, Nt)
    Ceq = H * (ppco2 + 2.3e-6/(60*60*24*360) * t)

    args = Ceq, K, Nt, Nz, a, dz, dt, kw

    C0 = Ceq[0] * np.ones(Nz)
    C = simulate(C0, args)
    plot_C(C, args)
    plot_M(C, args)


conv_test2()
# prob2()
# prob3()
# print(100/0.01)