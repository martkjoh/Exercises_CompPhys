import numpy as np
from scipy.sparse.linalg import inv, spsolve, splu
from scipy.sparse import diags, csc_matrix
from scipy.linalg import tri
from numba import njit

from matplotlib import pyplot as plt

plt.rcParams['mathtext.fontset'] = 'cm'
font = {'family' : 'serif', 
        'size': 20}
plt.rc('font', **font)
plt.rc('lines', lw=2)


def get_D(args):
    Ceq, K, T, N, a, dz, kw = args
    g = get_g(args)

    V0 = -4 * a  * K
    V0[0] += 2 * g

    V1 = np.zeros(N-1)
    V1[1:] = a * (K[2:] - K[:-2])/2 + 2 * a * K[1:-1]
    V1[0] = 4*a*K[0]

    V2 = np.zeros(N-1)
    V2[:-1] = -a * (K[2:] - K[:-2])/2 + 2 * a * K[1:-1]
    V2[-1] = 4*a*K[-1]

    return csc_matrix(diags((V2, V0, V1), (-1, 0, 1)))


def get_g(args):
    Ceq, K, T, N, a, dz, kw = args
    return 2*a*kw*dz/K[0] * (K[0] - 1/2 * (3/2 * K[0] + 2 * K[1] - 1/2 * K[2]))


def get_S(args):
    Ceq, K, T, N, a, dz, kw = args
    g = get_g(args)
    S = np.zeros((T, N))
    S[:, 0] = 2*g*Ceq
    return S

def tdma(A, b):
    @njit
    def tdma_solver(a, b, c, d):
        N = len(d)
        c_ = np.zeros(N-1)
        d_ = np.zeros(N)
        x  = np.zeros(N)
        c_[0] = c[0]/b[0]
        d_[0] = d[0]/b[0]
        for i in range(1, N-1):
            q = (b[i] - a[i-1]*c_[i-1])
            c_[i] = c[i]/q
            d_[i] = (d[i] - a[i-1]*d_[i-1])/q
        d_[N-1] = (d[N-1] - a[N-2]*d_[N-2])/(b[N-1] - a[N-2]*c_[N-2])
        x[-1] = d_[-1]
        for i in range(N-2, -1, -1):
            x[i] = d_[i] - c_[i]*x[i+1]
        return x
    x = tdma_solver(A.diagonal(-1), A.diagonal(0), A.diagonal(1), b)
    return x


def get_solve_tdma(args):
    Ceq, K, T, N, a, dz, kw = args
    D = get_D(args)
    I = csc_matrix(diags(np.ones(N), 0))
    A = I - D/2
    return lambda v: tdma(A, v)


def get_solve_spsolve(args):
    Ceq, K, T, N, a, dz, kw = args
    D = get_D(args)
    I = csc_matrix(diags(np.ones(N), 0))
    A = I - D/2
    return lambda v: spsolve(A, v) 


def get_solve_splu(args):
    Ceq, K, T, N, a, dz, kw = args
    D = get_D(args)
    I = csc_matrix(diags(np.ones(N), 0))
    A = I - D/2
    LU = splu(A)
    return lambda v: LU.solve(v)



def get_V(args):
    Ceq, K, T, N, a, dz, kw = args
    D = get_D(args)
    I = csc_matrix(diags(np.ones(N), 0))
    R =  I + D/2
    return lambda C, i, S: R.dot(C[i]) + (S[i] + S[i+1])/2
    

def simulate(C0, args, get_solve=get_solve_spsolve):
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


def plot_C(C, args):
    Ceq, K, T, N, a, dz, kw = args
    dt = a * dz**2 * 2
    extent = 0, T*dt, 0, N*dz
    
    fig, ax = plt.subplots(figsize=(12, 8))
    st = T//500+1
    sn = N//500+1
    im = ax.imshow(C[::st, ::sn].T, aspect="auto", extent=extent)
    fig.colorbar(im)
    ax.set_title("$K_0={},\,\\alpha={:.2f},\,k_w={}$".format(K[0], a, kw))
    plt.show()


def plot_D(args):
    D = get_D(args)
    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(D.todense())
    fig.colorbar(im)
    plt.show()


def plot_M(C, args):
    Ceq, K, T, N, a, dz, kw = args
    dt = a * dz**2 * 2
    t = np.linspace(0, T*dt, T)
    M = np.einsum("tz -> t", C)

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(t, M)
    ax.set_xlabel("$t$")
    ax.set_ylabel("$M$")
    plt.show()


def plot_var(C, args):
    Ceq, K, T, N, a, dz, kw = args
    dt = a * dz**2 * 2
    t = np.linspace(0, T*dt, T)
    z = np.linspace(0, N*dz, N)
    M = np.einsum("tz -> t", C) * dz
    mu = np.einsum("tz, z -> t", C, z) * dz / M
    v = np.einsum("z, t -> tz", z, -mu)
    var = np.einsum("tz, tz -> t", C, v**2) * dz / M

    m = np.max(var)
    lin = var[0] + K[0] * t / 2
    i = lin<m

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(t, var)
    ax.plot(t[i], lin[i], "--k")
    ax.set_xlabel("$t$")
    ax.set_ylabel("$\sigma^2$")
    plt.show()


def plot_M_decay(C, args):
    Ceq, K, T, N, a, dz, kw = args
    dt = a * dz**2 * 2
    L = dz * N
    t = np.linspace(0, T*dt, T)
    M = np.einsum("tz -> t", C)
    tau = L / kw * 2
    Bi = kw * L / np.min(K)
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(t, M)
    ax.plot(t, M[0] * np.exp(-t / tau), "--k")
    ax.set_title("$\mathrm{Bi}=" + str(Bi) + ",\, \\tau = " + str(tau) + "$")
    plt.show()


def get_args1(const_K=True):
    N = 1000
    T = 10000
    t0 = 0.0032
    dz = 1/N
    dt = t0*1/T
    a = dt / dz**2 / 2
    
    kw = 0
    K0 = 18
    if const_K:K = K0*np.ones(N)
    else: K = K0*(2 + np.sin(np.linspace(0, 10, N)))
    Ceq = np.ones(T)
    return Ceq, K, T, N, a, dz, kw

    
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
    C0 = np.exp(-(z - dz*N/2)**2/(2 * 1/100)**2)
    C = simulate(C0, args, get_solve_splu)
    plot_C(C, args)
    plot_M(C, args)
    plot_var(C, args)


def get_args2(const_K=True):
    N = 1000
    T = 100_000
    t0 = 1000
    dz = 1/N
    dt = t0/T
    a = dt / dz**2 / 2

    kw = 0.02
    K0 = 3100
    if const_K: K = K0 * np.ones(N)
    else: K = K0*(2 + np.sin(np.linspace(0, 10, N)))
    Ceq = np.zeros(T)
    return Ceq, K, T, N, a, dz, kw


def test4():
    args = get_args2(True)
    Ceq, K, T, N, a, dz, kw = args

    z = np.linspace(0, N * dz, N)
    C0 = np.ones(N)
    C = simulate(C0, args, get_solve_splu)
    plot_C(C, args)
    plot_M_decay(C, args)


# test1()
# test23()
test4()
