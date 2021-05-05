import numpy as np
from tqdm import trange
from utilities import integrate, integrate_untill, SIR_stoch, stoch_step, get_Nt


def get_test_stoch():
    N = 100_000
    I = 10
    x0 = np.array([N-I, I, 0], dtype=int)
    T = 180; dt = 0.1
    args = (0.25, 10) # beta, tau
    xs = []
    for i in trange(100):
        xs.append(integrate(SIR_stoch, x0, T, dt, args, step=stoch_step), inf=False)

    return xs, T, dt, args



def SIR_stochastic_convergence():
    N = 100_000
    I = 10
    x0 = np.array([N-I, I, 0], dtype=int)
    T = 200
    dts = [2, 1, 1/2, 1/2**2, 1/2**3, 1/2**4]
    runs = 100
    args = (0.25, 10) # beta, tau
    xs = []
    for i in trange(len(dts)):
        dt = dts[i]
        Nt = get_Nt(T, dt)
        x = np.zeros((Nt, *x0.shape))
        for _ in range(runs):
            x += integrate(SIR_stoch, x0, T, dt, args, step=stoch_step, inf=False)
        xs.append(x/runs)

    return xs, dts, args, T


def prob_disappear(run=False):
    # TODO: variance? Other statistical numbers
    datapath = "data/dissapear.npy"
    N = 100_000
    T = 20; dt = 0.1
    args = (0.25, 10) # beta, tau
    Is = np.arange(10+1)
    cond = lambda x: x[1]==0 or x[1]==100
    runs = 1000
    # How many times did the disease disappear?
    terminate = np.zeros((len(Is), runs))
    if run:
        for n in trange(len(Is)):
            I = Is[n]
            print("I = ", I)
            for i in trange(runs):
                x = np.array([N-I, I, 0], dtype=int)
                x, _, _ = integrate_untill(SIR_stoch, x, T, dt, args, cond, step=stoch_step, inf=False)
                terminate[n, i] = x[1]==0

        np.save(datapath, terminate)
    else:
        terminate = np.load(datapath)

    return terminate, Is



if __name__=="__main__":
    prob_disappear()