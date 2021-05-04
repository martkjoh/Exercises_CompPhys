import numpy as np
from tqdm import trange
from utillities import integrate, get_Nt, stoch_commute_step, SEIIaR_commute, get_pop_structure


def get_test_SEIIaR_commute():
    args = (0.55, 1, 0.1, 0.6, 0.4, 3, 7)
    N = np.array([
        [100_000, 0],
        [0, 1]
    ], dtype=int)
    E = np.array([
        [25, 0],
        [0, 0]
    ], dtype=int)
    Oh = np.zeros_like(N)
    x0 = np.array([N-E, E, Oh, Oh, Oh], dtype=int)
    T = 180; dt = 0.1
    xs = []
    for i in range(10):
        xs.append(integrate(
            SEIIaR_commute, x0, T, dt, args, step=stoch_commute_step)[:, :, 0, 0]
            )
    return xs, T, dt, args


def get_two_towns():
    args = (0.55, 1, 0.1, 0.6, 0.4, 3, 7)
    N = np.array([
        [9000, 1000],
        [200, 99800]
    ], dtype=int)
    E = np.array([
        [25, 0],
        [0, 0]
    ], dtype=int)
    Oh = np.zeros_like(N)
    # x[time, var, city_i, city_j]
    x0 = np.array([N-E, E, Oh, Oh, Oh], dtype=int)
    T = 180; dt = 0.1
    Nt = get_Nt(T, dt)
    xs = np.zeros((Nt, 5, 2, 2))
    for i in range(10):
        xs += integrate(SEIIaR_commute, x0, T, dt, args, step=stoch_commute_step)
    xs = np.sum(xs, axis=3)/10
    return xs, T, dt, args


def get_nine_towns():
    args = (0.55, 1, 0.1, 0.6, 0.4, 3, 7)
    N = np.array([
    [198600, 100, 100, 100, 100, 1000, 0, 0, 0, 0],
    [500, 9500, 0, 0, 0, 0, 0, 0, 0, 0],
    [500, 0, 9500, 0, 0, 0, 0, 0, 0, 0],
    [500, 0, 0, 9500, 0, 0, 0, 0, 0, 0],
    [500, 0, 0, 0, 9500, 0, 0, 0, 0, 0],
    [1000, 0, 0, 0, 0, 498200, 200, 200, 200, 200],
    [0, 0, 0, 0, 0, 1000, 0, 19000, 0, 0],
    [0, 0, 0, 0, 0, 1000, 0, 0, 19000, 0],
    [0, 0, 0, 0, 0, 1000, 0, 0, 19000, 0],
    [0, 0, 0, 0, 0, 1000, 0, 0, 0, 19000]]
    )
    E = np.zeros_like(N)
    E[1, 1] = 25
    Oh = np.zeros_like(N)
    x0 = np.array([N-E, E, Oh, Oh, Oh], dtype=int)
    T = 180; dt = 0.1
    Nt = get_Nt(T, dt)
    x = np.zeros((Nt, 5, *N.shape))
    for i in range(10):
        x += integrate(SEIIaR_commute, x0, T, dt, args, step=stoch_commute_step)
    x = np.sum(x, axis=3)/10
    return x, T, dt, args
    

def get_Norway():
    args = (0.55, 1, 0.1, 0.6, 0.4, 3, 7)
    N = get_pop_structure()
    E = np.zeros_like(N)
    E[0, 0] = 50
    Oh = np.zeros_like(N)
    x0 = np.array([N-E, E, Oh, Oh, Oh], dtype=int)
    T = 180; dt = .5
    save = 91
    x = np.zeros((save, 5, *N.shape))
    for i in trange(10):
        x += integrate(
            SEIIaR_commute, x0, T, dt, args, save=save, 
            step=stoch_commute_step, inf=False
            )
    x = np.sum(x, axis=3)/1
    return x, T, dt, args


def get_Norway_lockdown():
    args = (0.55, 1, 0.1, 0.6, 0.4, 3, 7)
    N = get_pop_structure(lockdown=True)
    E = np.zeros_like(N)
    E[0, 0] = 50
    Oh = np.zeros_like(N)
    x0 = np.array([N-E, E, Oh, Oh, Oh], dtype=int)
    T = 180; dt = .5
    save = 91
    x = np.zeros((save, 5, *N.shape))
    for i in range(10):
        x += integrate(
            SEIIaR_commute, x0, T, dt, args, save=save, 
            step=stoch_commute_step, inf=False
            )
    x = np.sum(x, axis=3)/1
    return x, T, dt, args



if __name__=="__main__":
    # get_test_SEIIaR_commute()
    # get_two_towns()
    # get_pop_structure()
    # get_Norway()
    pass