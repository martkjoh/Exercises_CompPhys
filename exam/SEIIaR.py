import numpy as np
from utilities import integrate, get_Nt, stoch_step, SEIIaR
from tqdm import trange
    

def get_test_SEIIAR():
    #       beta, rs, ra, fs, fa, tE, tI
    args = (0.55, 1, 0.1, 0.6, 0.4, 3, 7)
    N = 100_000
    E = 25
    x0 = np.array([N-E, E, 0, 0, 0], dtype=int)
    T = 180; dt = 0.1
    xs = []
    for i in trange(10):
        xs.append(integrate(SEIIaR, x0, T, dt, args, step=stoch_step, inf=False))

    return xs, T, dt, args


def SEIIaR_convergence():
    N = 100_000
    E = 25
    x0 = np.array([N-E, E, 0, 0, 0], dtype=int)
    T = 180
    dts = [2, 1, 1/2, 1/2**2, 1/2**3, 1/2**5]
    args = (0.55, 1, 0.1, 0.6, 0.4, 3, 7)
    runs = 100
    xs = []
    for i in trange(len(dts)):
        
        dt = dts[i]
        Nt = get_Nt(T, dt)
        x = np.zeros((Nt, *x0.shape))
        for _ in range(runs):
            x += integrate(SEIIaR, x0, T, dt, args, step=stoch_step, inf=False)
        xs.append(x/runs)

    return xs, dts, args, T


def stay_home(run=False):
    datapath = "data/stay_home.npy"
    runs = 101
    rss = np.linspace(0, 1, runs)
    samples = 100
    N = 100_000
    E = 25
    x0 = np.array([N-E, E, 0, 0, 0], dtype=int)
    T = 20; dt = 0.1
    save = 101
    xs = np.zeros((samples, runs, save, len(x0)), dtype=type(x0))
    if run:
        for i in trange(runs):
            rs = rss[i]
            args = (0.55, rs, 0.1, 0.6, 0.4, 3, 7)
            for j in range(samples):
                xs[j, i] = integrate(
                    SEIIaR, x0, T, dt, args, step=stoch_step, inf=False, save=save
                    )

        np.save(datapath, xs)
    else:
        xs = np.load(datapath, allow_pickle=True)


    return xs, T, dt, rss


if __name__=="__main__":
    # get_test_SEIIAR()
    stay_home()