import numpy as np
from numpy import cos, sin, exp, pi
from tqdm import trange
from mayavi import mlab
import ffmpeg
import os

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
        NNsum += np.roll(S, 1, i) + np.roll(S, -1, i)
    return NNsum


def get_H(S, J, dz, B):
    """ returns the field """
    NNsum = NN(S)
    aniso = np.zeros_like(S)
    aniso[:, :, :, 2] = S[:, :, :, 2]
    return J * NNsum + 2*dz*aniso + B 


def LLG(S, J, dz, B, a):
    H = get_H(S, J, dz, B)
    dtS = np.einsum("...ac, ...c-> ...a", np.einsum("abc, ...b -> ...ac", eijk, S), H)
    if a:
        sum1 = np.einsum("...b, ...b -> ...", S, S)
        sum2 = np.einsum("...b, ...b -> ...", S, H)
        sum1 = np.einsum("...j, ...ji -> ...ji", sum1, H)
        sum2 = np.einsum("...j, ...ji -> ...ji", sum2, S)
        dtS += a * (sum1 - sum2)
    return dtS


def integrate(f, S, h, step, args):
    for n in trange(len(S)-1):
        step(f, S, h, n, args)


def random(n):
    theta = np.random.random(n**dim).reshape(n, n, n) * pi
    phi = np.random.random(n**dim).reshape(n, n, n) * 2 * pi
    return np.array([cos(phi)*sin(theta), sin(phi)*sin(theta), cos(theta)]).T


def get_comp(S, t, i):
    return S[t, :, :, :, i]


path = "plots/"
name = "test_anim"
ext = ".png"

def anim_spins(S, skip=1):
    T = len(S)
    N = len(S[0])
    l = N/2
    x, y, z= np.mgrid[-l:l:N*1j, -l:l:N*1j, -l:l:N*1j]
    fig = mlab.figure(size=(1_000, 1_000))
    quiver = mlab.quiver3d(
        x, y, z, get_comp(S, 0, 0), get_comp(S, 0, 1), get_comp(S, 0, 2),
        mode="arrow", resolution=16, scalars=np.arange(N**dim).reshape(N, N, N), 
        colormap="plasma")
    quiver.glyph.color_mode = "color_by_scalar"
    
    pad = len(str(N))
    @mlab.animate(delay=50)
    def anim():
        for i in range(T//skip):
            quiver.mlab_source.u = get_comp(S, i*skip, 0)
            quiver.mlab_source.v = get_comp(S, i*skip, 1)
            quiver.mlab_source.w = get_comp(S, i*skip, 2)

            zeros = '0'*(pad - len(str(i)))
            filename = path + name + "_{}{}".format(zeros, i) + ext
            mlab.savefig(filename)
            yield
        
        mlab.clf()
        mlab.close(all=True)


    anim()
    mlab.show()


    input = ffmpeg.input(path + name + "_%0" + str(len(str(N))) + "d.png")
    output = path + name + ".mp4"
    stream = ffmpeg.output(input, output, framerate=20)

    if os.path.isfile(output): os.remove(output)
    ffmpeg.run(stream)  

    [os.remove(path + f) for f in os.listdir(path) if f.endswith(".png") and f[:len(name)]==name]


T, N, h = 10_000, 30, 0.01
S = np.empty([T] + [N] * dim + [dim])
S[0] = random(N)

args = (1, 0.1, [0, 0, 0], 0.05) # (J, dz, B, a)

integrate(LLG, S, h, heun_step, args)
anim_spins(S, 10)

