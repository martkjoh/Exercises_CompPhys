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


def stay_home():
    runs = 100
    rss = np.linspace(0, 1, runs)
    samples = 100
    N = 100_000
    E = 25
    x0 = np.array([N-E, E, 0, 0, 0], dtype=int)
    T = 20; dt = 0.1
    Nt = get_Nt(T, dt)
    xs = np.zeros((runs, Nt, len(x0)), dtype=type(x0))
    for i in trange(runs):
        rs = rss[i]
        args = (0.55, rs, 0.1, 0.6, 0.4, 3, 7)
        for _ in range(samples):
            xs[i] += integrate(SEIIaR, x0, T, dt, args, step=stoch_step, inf=False)
    
    xs *= 1/samples
    xs = np.array(xs, dtype=np.float64)
    
    # the growth seems to start approx after 5 days
    n = int(5/dt) # place to start measuring from
    logE = np.log(xs[:, n:, 1])
    av_growth = (logE[:, -1] - logE[:, 0]) / (T - dt*n)
    # The index of the highest v with positive growth rate
    high_i = np.arange(0, len(rss))[np.less(av_growth, 0)][-1]
    print("Greates rs without exp growth: {}".format(rss[high_i]))
    print("Corr growth rate: {}".format(av_growth[high_i]))
    print("Reach at index {} of {}".format(high_i, len(rss)))
    print("Lowest rs value yielding exp growth: {}".format(rss[high_i+1]))

    return xs, T, dt, args, rss, av_growth, high_i


if __name__=="__main__":
    get_test_SEIIAR()