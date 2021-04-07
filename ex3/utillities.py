import numpy as np
from numpy import newaxis as na
from scipy.sparse.linalg import splu
from scipy.sparse import diags, csc_matrix
from scipy.integrate import simpson
from tqdm import trange

"""
Sampling functions
"""


def get_tz(C, args):
    Ceq, K, Nt, Nz, a, dz, dt, kw, L, t0 = args
    Nt, Nz = C.shape
    t = np.linspace(0, t0, Nt)
    z = np.linspace(0, L, Nz)
    return t, z


def get_mass(C, args):
    Ceq, K, Nt, Nz, a, dz, dt, kw, L, t0 = args
    return simpson(C, dx=dz, axis=1)


def get_var(C, args):
    Ceq, K, Nt, Nz, a, dz, dt, kw, L, t0 = args
    t, z = get_tz(C, args)
    M = get_mass(C, args)
    mu = simpson(C * z[na,:], dx=dz, axis=1) / M
    v2 = (z[na, :] -mu[:, na])**2
    var = simpson(C * v2, dx=dz, axis=1) / M
    return var


def get_rms(Cs, Nzs=None):
    if not(Nzs is None): 
        skips = [int((Nzs[-1]-1)/(Nzs[i]-1)) for i in range(len(Nzs))]
    else: 
        skips = [1 for _ in Cs]
    rms = lambda C, C0: np.sqrt(np.mean(((C-C0)/C0)**2))
    return [rms(Cs[i], Cs[-1][::skips[i]]) for i in range(len(Cs)-1)]


"""
Utillities for full implementation
"""


def get_D(args):
    Ceq, K, Nt, Nz, a, dz, dt, kw, L, t0 = args
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
    Ceq, K, Nt, Nz, a, dz, dt, kw, L, t0 = args
    A = 2*a*kw*dz
    B =  1 - (-3/2 * K[0] + 2 * K[1] - 1/2 * K[2]) /(2 * K[0])
    return A * B


def get_S(Ceqi, g, args):
    Ceq, K, Nt, Nz, a, dz, dt, kw, L, t0 = args
    S = np.zeros((Nz))
    S[0] = 2*g*Ceqi
    return S


def get_solve_V(args):
    Ceq, K, Nt, Nz, a, dz, dt, kw, L, t0 = args
    D = get_D(args)
    I = csc_matrix(diags(np.ones(Nz), 0))
    A = I - D/2
    LU = splu(A)    
    R =  I + D/2
    return lambda v: LU.solve(v), lambda C, Si, Sip: R.dot(C) + (Si + Sip)/2


def simulate(C0, args):
    Ceq, K, Nt, Nz, a, dz, dt, kw, L, t0 = args
    C = np.zeros((Nt, Nz))
    C[0] = C0
    g = get_g(args)
    solve, V = get_solve_V(args)

    for i in trange(Nt-1):
        Si = get_S(Ceq[i], g, args)
        Si1 = get_S(Ceq[i+1], g, args)
        vi = V(C[i], Si, Si1)
        C[i + 1] = solve(vi)

    return C


def simulate_until(C0, args):
    Ceq, K, Nt, Nz, a, dz, dt, kw, L, t0 = args
    C = C0
    g = get_g(args)
    solve, V = get_solve_V(args)

    for i in trange(Nt-1):
        Si = get_S(Ceq[i], g, args)
        Si1 = get_S(Ceq[i+1], g, args)
        vi = V(C, Si, Si1)
        C = solve(vi)

    return C


def get_Nzs(n, Nz_max):
    assert (Nz_max - 1)%5 == 0
    Nzs = [Nz_max, (Nz_max-1)//5 + 1]
    for i in range(n):
        Nz = Nzs[i+1]
        assert (Nz - 1)%2 == 0
        Nzs.append((Nz-1)//2 + 1)
    return np.array(Nzs[::-1], dtype=int)
