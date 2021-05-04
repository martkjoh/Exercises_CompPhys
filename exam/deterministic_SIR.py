import numpy as np
from tqdm import trange
from utilities import integrate, SIR



def get_testSIR():
    eps = 1e-4
    x0 = np.array([1-eps, eps, 0])
    T = 180; dt = 0.01
    args = (0.25, 10) # beta, tau
    x = integrate(SIR, x0, T, dt, args)
    return x, T, dt, args


def flatten_the_curve():
    eps = 1e-4
    x0 = np.array([1-eps, eps, 0])
    T = 180; dt = 0.01
    betas = np.linspace(0.15, 0.25, 100)
    max_I = []
    max_day = []
    for i in trange(len(betas)):
        b = betas[i]
        args = (b, 10) # beta, tau
        x = integrate(SIR, x0, T, dt, args, inf=False)
        max_I.append(np.max(x[:, 1]))
        max_day.append(np.argmax(x[:, 1]))

    # The index of the highest beta above 0.2
    high_i = np.arange(0, len(betas))[np.less(max_I, 0.2)][-1]
    print("Lates day to reach top = {}".format(np.max(max_day)))
    print("Highest beta giving I below 0.2: {}".format(betas[high_i]))
    print("Highest I below 0.2: {}".format(max_I[high_i]))
    print("Reach at index {} of {}".format(high_i, len(betas)))

    return max_I, betas, high_i


def vaccination():
    eps = 1e-4
    T = 1; dt = 0.1
    vacc = np.linspace(0, 1, 100)
    args = (0.25, 10) # beta, tau
    xs = []
    for i in trange(len(vacc)):
        v = vacc[i]
        x0 = np.array([1-eps-v, eps, v])
        xs.append(integrate(SIR, x0, T, dt, args, inf=False))

    growth_rate = [np.log(x[1, 1]/x[0, 1]) for x in xs]
    # The index of the highest v with positive growth rate
    high_i = np.arange(0, len(vacc))[np.greater(growth_rate, 0)][-1]
    print("highest v with positive growth rate: {}".format(vacc[high_i]))
    print("Corr growth rate: {}".format(growth_rate[high_i]))
    print("Reach at index {} of {}".format(high_i, len(vacc)))
    return growth_rate, vacc, high_i



if __name__=="__main__":
    # flatten_the_curve()    
    vaccination()
