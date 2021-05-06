import numpy as np
from tqdm import trange
from utilities import SEIIaR_commute2, integrate, get_Nt, stoch_commute_step, SEIIaR_commute, get_pop_structure


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
    for i in trange(10):
        xs.append(integrate(
            SEIIaR_commute, x0, T, dt, args, step=stoch_commute_step, inf=False
            )[:, :, 0, 0])
    return xs, T, dt, args


def SEIIaR_commute_convergence(run=False):
    datapath = "data/commuter_conv.npy"
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
    T = 180
    dts = [2, 1, 1/2, 1/2**2, 1/2**3, 1/2**5]
    args = (0.55, 1, 0.1, 0.6, 0.4, 3, 7)
    runs = 100
    xs = []
    if run:
        for i in trange(len(dts)):
            dt = dts[i]
            Nt = get_Nt(T, dt)
            x = np.zeros((Nt, 5))
            for _ in range(runs):
                x += integrate(
                    SEIIaR_commute, x0, T, dt, args, step=stoch_commute_step, inf=False
                    )[:, :, 0, 0]
            xs.append(x/runs)

        np.save(datapath, np.array(xs))
    else:
        xs = np.load(datapath, allow_pickle=True)

    return xs, dts, args, T


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
    for i in trange(10):
        xs += integrate(
            SEIIaR_commute, x0, T, dt, args, step=stoch_commute_step, inf=False
            )
    xs = np.sum(xs, axis=3)/10
    return xs, T, dt, args


def get_two_towns2():
    args = (0.55, 1, 0.1, 0.6, 0.4, 3, 7)
    N = np.array([
        [0, 10000],
        [10000, 0]
    ], dtype=int)
    E = np.array([
        [0, 25],
        [0, 0]
    ], dtype=int)
    Oh = np.zeros_like(N)
    # x[time, var, city_i, city_j]
    x0 = np.array([N-E, E, Oh, Oh, Oh], dtype=int)
    T = 180; dt = 0.1
    Nt = get_Nt(T, dt)
    xs = np.zeros((Nt, 5, 2, 2))
    for i in trange(10):
        xs += integrate(
            SEIIaR_commute, x0, T, dt, args, step=stoch_commute_step, inf=False
            )
    xs = np.sum(xs, axis=3)/10
    return xs, T, dt, args



def get_nine_towns():
    args = (0.55, 1, 0.1, 0.6, 0.4, 3, 7)
    N = np.array([
    [198600, 100,  100,  100,  100,  1000,   0,     0,     0,     0     ],
    [500,    9500, 0,    0,    0,    0,      0,     0,     0,     0     ],
    [500,    0,    9500, 0,    0,    0,      0,     0,     0,     0     ],
    [500,    0,    0,    9500, 0,    0,      0,     0,     0,     0     ],
    [500,    0,    0,    0,    9500, 0,      0,     0,     0,     0     ],
    [1000,   0,    0,    0,    0,    498200, 200,   200,   200,   200   ],
    [0,      0,    0,    0,    0,    1000,   19000, 0,     0,     0     ],
    [0,      0,    0,    0,    0,    1000,   0,     19000, 0,     0     ],
    [0,      0,    0,    0,    0,    1000,   0,     0,     19000, 0     ],
    [0,      0,    0,    0,    0,    1000,   0,     0,     0,     19000 ]
    ])
    E = np.zeros_like(N)
    E[1, 1] = 25
    Oh = np.zeros_like(N)
    x0 = np.array([N-E, E, Oh, Oh, Oh], dtype=int)
    T = 180; dt = 0.1
    Nt = get_Nt(T, dt)
    x = np.zeros((Nt, 5, *N.shape))
    for i in trange(10):
        x += integrate(
            SEIIaR_commute, x0, T, dt, args, step=stoch_commute_step, inf=False
            )
    x = np.sum(x, axis=3)/10
    return x, T, dt, args
    

def get_Norway(datapath, lockdown=False, run=False):
    args = (0.55, 1, 0.1, 0.6, 0.4, 3, 7)
    N = get_pop_structure(lockdown)
    E = np.zeros_like(N)
    E[0, 0] = 50
    Oh = np.zeros_like(N)
    x0 = np.array([N-E, E, Oh, Oh, Oh], dtype=int)
    T = 180; dt = .5
    save = 121
    runs = 2

    if run:
        xs = np.empty((runs, save, 5, *N.shape))
        for i in trange(runs):
            xs[i] = integrate(
                SEIIaR_commute, x0, T, dt, args, save=save, 
                step=stoch_commute_step, inf=True
                )
        xs = np.sum(xs, axis=4)
        np.save(datapath, xs)
    else:
        xs = np.load(datapath)
    return xs, T, dt, args


# Functions for profiling purposes

def prof1():
    args = (0.55, 1, 0.1, 0.6, 0.4, 3, 7)
    N = get_pop_structure()
    E = np.zeros_like(N)
    E[0, 0] = 50
    Oh = np.zeros_like(N)
    x0 = np.array([N-E, E, Oh, Oh, Oh], dtype=int)
    T = 180; dt = .1
    save = 121
    integrate(SEIIaR_commute, x0, T, dt, args, save=save, step=stoch_commute_step)

def prof2():
    args = (0.55, 1, 0.1, 0.6, 0.4, 3, 7)
    N = get_pop_structure()
    E = np.zeros_like(N)
    E[0, 0] = 50
    Oh = np.zeros_like(N)
    x0 = np.array([N-E, E, Oh, Oh, Oh], dtype=int)
    T = 180; dt = .1
    save = 121
    integrate(SEIIaR_commute2, x0, T, dt, args, save=save, step=stoch_commute_step)
    




if __name__=="__main__":
    # get_test_SEIIaR_commute()
    # get_two_towns()
    # get_pop_structure()
    # get_Norway()
    
    # N = ge
    # t_pop_structure()
    # Nl = get_pop_structure(lockdown=True)
    # print(N.shape)
    # pop = np.sum(N, axis=1).astype(int)
    # popl = np.sum(Nl, axis=1).astype(int)
    # print(pop-popl)

    # print(N.size)
    # print(np.sum(N>0))

    # print(np.sum(N[0].astype(int))) # Working populace, Oslo

    # i2 = np.argmax(pop[1:]) + 1
    # print(i2)
    # print(pop[i2])
    # print(N[0, i2])
    # print(N[i2, 0])
    
    
    prof2()
    pass