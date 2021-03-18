import numpy as np
from scipy.linalg import inv
from scipy.sparse import diags
from matplotlib import pyplot as plt


def get_D(args):
    Ceq, K, T, N, a, dz, kw = args
    g = get_g(args)

    V0 = -4 * a  * K
    V0[0] += 2 * g * K[0]

    V1 = np.zeros(N-1)
    V1[1:] = a * (K[2:] - K[:-2])/2 + 2 * a * K[1:-1]
    V1[0] = 4*a*K[0]

    V2 = np.zeros(N-1)
    V2[:-1] = -a * (K[2:] - K[:-2])/2 + 2 * a * K[1:-1]
    V2[-1] = 4*a*K[-1]

    return diags((V2, V0, V1), (-1, 0, 1))


def get_g(args):
    Ceq, K, T, N, a, dz, kw = args
    return 2*a*kw*dz/K[0] * (K[0] - 1/2 * (3/2 * K[0] + 2 * K[1] - 1/2 * K[2]))


def get_S(args):
    Ceq, K, T, N, a, dz, kw = args
    g = get_g(args)
    S = np.zeros((T, N))
    S[:, 0] = 2*g*Ceq
    return S


def get_solve(args):
    Ceq, K, T, N, a, dz, kw = args
    D = get_D(args).todense()
    A_inv = inv(np.identity(N) - D/2)
    return lambda v: A_inv @ v


def get_V(args):
    Ceq, K, T, N, a, dz, kw = args
    D = get_D(args).todense()
    R =  np.identity(N) + D/2
    return lambda C, i, S: R @ C[i] + (S[i] + S[i+1])/2
    

def simulate(C0, args):
    Ceq, K, T, N, a, dz, kw = args
    C = np.zeros((T, N))
    C[0] = C0
    S = get_S(args)

    solve = get_solve(args)
    V = get_V(args)

    for i in range(T-1):
        vi = V(C, i, S).T
        C[i + 1] = solve(vi).T

    return C

def get_args1(const_K=True):
    N = 1000
    T = 1000
    dz = 0.1
    a = 1
    kw = 0
    if const_K:K = np.ones(N)
    else: K = 2 + np.sin(np.linspace(0, 10, N))
    Ceq = np.ones(T)
    return Ceq, K, T, N, a, dz, kw


def plot_C(C, args):
    Ceq, K, T, N, a, dz, kw = args
    dt = a * dz**2 * 2
    t = np.linspace(0, T*dt, T)
    z = np.linspace(0, N*dz, N)
    t, z = np.meshgrid(t, z)
    fig, ax = plt.subplots()
    extent = 0, T*dt, 0, N*dz
    im = ax.imshow(C.T)
    fig.colorbar(im)
    plt.show()


def plot_D():
    args = get_args1()

    D = get_D(args)
    fig, ax = plt.subplots()
    im = ax.imshow(D.todense())
    fig.colorbar(im)
    plt.show()

def plot_M(C, args):
    Ceq, K, T, N, a, dz, kw = args
    dt = a * dz**2 * 2
    t = np.linspace(0, T*dt, T)
    M = np.einsum("tz -> t", C)

    fig, ax = plt.subplots()
    ax.plot(t, M)
    plt.show()

def plot_var(C, args):
    Ceq, K, T, N, a, dz, kw = args
    dt = a * dz**2 * 2
    t = np.linspace(0, T*dt, T)
    z = np.linspace(0, N*dz, N)
    M = np.einsum("tz -> t", C) * dz
    mu = np.einsum("tz, z -> t", C, z) * dz / M
    var = np.einsum("tz, z -> t", C, (z - mu)**2) * dz / M

    fig, ax = plt.subplots()
    ax.plot(t, var)
    plt.show()


def test1():
    args = get_args1()
    Ceq, K, T, N, a, dz, kw = args

    C0 = np.ones(N)
    C = simulate(C0, args)
    plot_C(C, args)


def test23():
    args = get_args1()
    Ceq, K, T, N, a, dz, kw = args

    z = np.linspace(0, N * dz, N)
    C0 = np.exp(- (z - dz*N/2)**2/5)
    C = simulate(C0, args)
    plot_C(C, args)
    plot_M(C, args)
    plot_var(C, args)


def get_args2(const_K=True):
    N = 1000
    T = 1000
    dz = 0.1
    a = 1
    kw = 1
    if const_K:K = np.ones(N)
    else: K = 2 + np.sin(np.linspace(0, 10, N))
    Ceq = np.ones(T)
    return Ceq, K, T, N, a, dz, kw


def test4()