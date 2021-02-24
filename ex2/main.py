import numpy as np
import matplotlib.pyplot as plt
from numpy import cos, sin


dim = 3


eijk = np.zeros((3, 3, 3))
eijk[0, 1, 2] = eijk[1, 2, 0] = eijk[2, 0, 1] = 1
eijk[0, 2, 1] = eijk[2, 1, 0] = eijk[1, 0, 2] = -1


def heun_step(f, y, h, n, args):
    y[n+1] = y[n] + h * f(y[n], *args)
    y[n+1] = y[n] + (h / 2) * (f(y[n], *args) + f(y[n+1], *args))


def NN(S):
    NNsum = np.zeros_like(S)
    for i in range(dim):
        NNsum = np.roll(S, 1, i) + np.roll(S, -1, i)
    return NNsum


def get_H(S, dz, B):
    """ returns the field """
    NNsum = NN(S)
    aniso = np.zeros_like(S)
    aniso[:, :, :, 2] = S[:, :, :, 2]
    return NNsum + 2*dz*aniso + B 


def LLG(S, dz, B, a):
    H = get_H(S, dz, B)
    dtS = np.einsum("...ac, ...c-> ...a", np.einsum("abc, ...b -> ...ac", eijk, S), H)
    if a:
        sum1 = np.einsum("...b, ...b -> ...", S, S) * H
        sum2 = np.einsum("...b, ...b -> ...", S, H) * S
        dtS += a * (sum1 - sum2)
    return dtS


def integrate(f, S, h, step, args):
    for n in range(T-1):
        step(f, S, h, n, args)



T = 100
N = 1
S = np.empty([T] + [N] * dim + [dim])

theta = 0.4
phi = 0
S[0, 0, 0, 0] = np.array([cos(phi)*sin(theta), sin(phi)*sin(theta), cos(theta)])

h = 0.1
a = 0
dz = 0
B = np.array([0, 0, 1])
args = (dz, B, a)

integrate(LLG, S, h, heun_step, args)
t = np.linspace(0, T*h, T)


coo = ["x", "y", "z"]
fig, ax = plt.subplots()
for i in range(dim):
    ax.plot(S[:, 0, 0, 0, i], label="$S_"+coo[i]+"$")

ax.legend()
plt.show()