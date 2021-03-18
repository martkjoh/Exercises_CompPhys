import numpy as np
from scipy.linalg import inv
from scipy.sparse import diags
from matplotlib import pyplot as plt


def get_D(K, a, g):
    N = len(K)

    V0 = 4 * a  * K
    V0[0] += 2 * g

    V1 = np.zeros(N-1)
    V1[1:] = -a * (K[:-2] - K[2:])/2 - 2 * a * K[1:]
    V1[0] = -4*a*K[0]

    V2 = np.zeros(N-1)
    V2[:-1] = a * (K[:-2] - K[2:])/2 - 2 * a * K[1:]
    V2[-1] = 4*a*K[-1]

    return diags((V2, V0, V1), (-1, 0, 1))


def get_R():


def get_g(args):
    Ceq, K, T, N, a, dz, kw = args
    return 2*a*kw*dz/K[0] * (K[0] - 1/2 * (3/2 * K[0] + 2 * K[1] * 1/2 * K[2]))


def get_v(C, i, R, S):
    return R @ C[i] + (S[i] + S[i+1])/2


def solve(L_inv, V):
    return L_inv @ V


def step(C, i, R, S, solve):
    v = get_v(C, i, R, S)
    C[i + 1] = solve(C[i])


def simulate(C0, args):
    Ceq, K, T, N, a, dz, kw = args
    C = np.array((N, T))
    C[0] = C0
    g = get_g(args)
    D = get_D(K, a, g)


def test1():
    N = 1000

