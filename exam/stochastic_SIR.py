import numpy as np
from tqdm import trange
from utillities import integrate, integrate_untill, SIR_stoch, stoch_step


def get_test_stoch():
    N = 100_000
    I = 10
    x0 = np.array([N-I, I, 0], dtype=int)
    T = 180; dt = 0.1
    args = (0.25, 10) # beta, tau
    xs = []
    for i in range(100):
        xs.append(integrate(SIR_stoch, x0, T, dt, args, step=stoch_step))

    return xs, T, dt, args



def prob_disappear():
    # TODO: variance? Other statistical numbers
    N = 100_000
    T = 20; dt = 0.1
    args = (0.25, 10) # beta, tau
    Is = np.arange(10+1)
    cond = lambda x: x[1]==0 or x[1]==100
    # How many times did the disease disappear?
    terms = np.zeros_like(Is, dtype=float)
    runs = 1000
    for n in trange(len(Is)):
        I = Is[n]

        print("I = ", I)
        for _ in trange(runs):
            x = np.array([N-I, I, 0], dtype=int)
            x, _, _ = integrate_untill(SIR_stoch, x, T, dt, args, cond, step=stoch_step, inf=False)
            terms[n] += x[1]==0

        terms[n] = terms[n]/runs
    
    return terms, Is



if __name__=="__main__":
    prob_disappear()