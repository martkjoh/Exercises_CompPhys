import numpy as np
from utillities import *
from plot import *


H = 5060
kw = 5.97e-5

def prob2():
    def get_K(z, L):
        
        K0 = 1e-3
        Ka = 2e-2
        za = 7
        Kb = 5e-2
        zb = 10
        return K0 + Ka * z / za  *np.exp(z / za) + Kb * (L - z)/zb * np.exp(-(L - z)/zb)

    t0 = 60*60*24 * 180

    Nz = 10_000
    Nt = 10_000
    dz = 100/Nz
    dt = t0/Nt
    a = dt / (2*dz**2)

    z = np.linspace(0, Nz*dz, Nz)
    K = get_K(z, Nz*dz)

    ppco2 = 415e-6
    Ceq = H * ppco2

    args = Ceq, K, Nt, Nz, a, dz, dt, kw

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

    ppco2 = 415e-6
    Ceq = H * ppco2

    args = Ceq, K, Nt, Nz, a, dz, dt, kw

    C0 = np.zeros(Nz)
    C = simulate(C0, args)
    plot_C(C, args)
    plot_M(C, args)


# prob2()
prob3()
