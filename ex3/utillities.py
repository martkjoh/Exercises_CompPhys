import numpy as np
from numpy import newaxis as na
from scipy.sparse.linalg import spsolve, splu
from scipy.sparse import diags, csc_matrix
from scipy.integrate import simps
from numba import jit


"""
Sampling functions
"""


def get_tz(C, args):
    Ceq, K, Nt, Nz, a, dz, dt, kw = args
    L, t0 = Nz*dz, Nt*dt
    Nt, Nz = len(C), len(C[0])
    t = np.linspace(0, t0, Nt)
    z = np.linspace(0, L, Nz)
    return t, z


def get_mass(C, args):
    Ceq, K, Nt, Nz, a, dz, dt, kw = args
    t, z = get_tz(C, args)
    return simps(C, x=z, axis=1)


def get_var(C, args):
    M = get_mass(C, args)
    t, z = get_tz(C, args)
    mu = simps(C * z[na,:], x=z, axis=1) / M
    v = np.einsum("z, t -> tz", z, -mu)
    var = simps(C * v**2, x=z, axis=1) / M
    return var


"""
Utillities for full implementation
"""


def get_D(args):
    Ceq, K, Nt, Nz, a, dz, dt, kw = args
    g = get_g(args)

    V0 = -4*a*K
    V0[0] += -2 * g

    V1 = np.zeros(Nz-1)
    V1[1:] = a * (K[2:] - K[:-2])/2 + 2 * a * K[1:-1]
    V1[0] = 4*a*K[0]

    V2 = np.zeros(Nz-1)
    V2[:-1] = -a * (K[2:] - K[:-2])/2 + 2 * a * K[1:-1]
    V2[-1] = 4*a*K[-1]

    return csc_matrix(diags((V2, V0, V1), (-1, 0, 1)))


def get_g(args):
    Ceq, K, Nt, Nz, a, dz, dt, kw = args
    A = 2*a*kw*dz
    B =  1 - (-3/2 * K[0] + 2 * K[1] - 1/2 * K[2]) /(2 * K[0])
    return A * B


def get_S(args):
    Ceq, K, Nt, Nz, a, dz, dt, kw = args
    g = get_g(args)
    S = np.zeros((Nt, Nz))
    S[:, 0] = 2*g*Ceq
    return S


def get_S_const(args):
    Ceq, K, Nt, Nz, a, dz, dt, kw = args
    g = get_g(args)
    S = np.zeros((Nz))
    S[0] = 2*g*Ceq
    return S


def is_equib(C, args):
    Ceq, K, Nt, Nz, a, dz, dt, kw = args
    return np.allclose(C, Ceq)


def get_solve_V(args):
    Ceq, K, Nt, Nz, a, dz, dt, kw = args
    D = get_D(args)
    I = csc_matrix(diags(np.ones(Nz), 0))
    A = I - D/2
    LU = splu(A)    
    R =  I + D/2
    return lambda v: LU.solve(v), lambda C, Si, Sip: R.dot(C) + (Si + Sip)/2


def simulate(C0, args):
    Ceq, K, Nt, Nz, a, dz, dt, kw = args
    C = np.zeros((Nt, Nz))
    C[0] = C0
    S = get_S_const(args)
    solve, V = get_solve_V(args)

    for i in range(Nt-1):
        vi = V(C[i], S, S)
        C[i + 1] = solve(vi)

    return C

def simulate_until(C0, args, cond):
    Ceq, K, Nt, Nz, a, dz, dt, kw = args
    C = C0
    S = get_S_const(args)
    solve, V = get_solve_V(args)

    i = 0
    while not cond(C, args) and i<Nt:
        vi = V(C[i], S, S).T
        C = solve(vi).T

    return C
