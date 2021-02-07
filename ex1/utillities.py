import heapq
import time
import numpy as np
import os
from os import getcwd, path, mkdir
from progress.bar import Bar

if not os.getcwd().split("/")[-1] == "ex1":
    os.chdir(os.getcwd() + "/ex1")


# Sides of the box
L = 1 

"""
utillities
"""

def get_next_col(collisions):
    return heapq.heappop(collisions)


def push_next_collision(particles, n, i, t, collisions, radii):
    wall0 = check_wall_collison(particles[n, i, 0], particles[n, i, 2], radii[i])
    wall1 = check_wall_collison(particles[n, i, 1], particles[n, i, 3], radii[i])
    if wall0 != np.inf: heapq.heappush(collisions, (t+wall0, i, -1, t, "wall0"))
    if wall1 != np.inf: heapq.heappush(collisions, (t+wall1, i, -1, t, "wall1"))
    for j in range(len(particles[0])):
        dt = check_particle_collision(particles, n, i, j, radii)
        if dt != np.inf: heapq.heappush(collisions, (t+dt, i, j, t, "particle"))


def init_collisions(particles, radii):
    collisions = []
    for i in range(len(particles[0])):
        push_next_collision(particles, 0, i, 0, collisions, radii)
    return collisions


def simulate(dir_path, kwargs):
    random_dist, N, T, radii, masses, xi, xi_p = kwargs
    particles, t = run_loop(random_dist, N, T, radii, masses, xi, xi_p)
    print(os.getcwd())
    if not path.isdir(dir_path):
        mkdir(dir_path)
    np.save(dir_path + "particles.npy", particles)
    np.save(dir_path + "t.npy", t)


def read_data(path):
    particles = np.load(path + "particles.npy")
    t = np.load(path + "t.npy")
    return particles, t


"""
Physics
"""

def get_vel2(particles, n):
    return np.einsum("ij -> i", particles[n, :, 2:]**2)


def get_energy(particles, masses, n):
    return 1/2 * masses @ (get_vel2(particles, n))


def get_temp(particles, masses, n, N):
    return get_energy(particles, masses, n) / N


def MaxBoltz(v, m, T):
    return m * v / T * np.exp(- m * v**2 / (2 * T))


def check_wall_collison(x, v, r):
    dt = np.inf
    if v > 0:
        dt = (L - r - x) / v
    elif v < 0:
        dt = (r - x) / v
    return dt


def check_particle_collision(particles, n, i, j, radii):
    R = radii[i] + radii[j]
    dx = particles[n, j, :2] - particles[n, i, :2]
    dv = particles[n, j, 2:] - particles[n, i, 2:]
    d = (dv @ dx)**2 - (dv @ dv) * ((dx @ dx) - R**2)
    if (d <= 0 or dv @ dx >= 0): return np.inf
    else: return - (dv @ dx + np.sqrt(d)) / (dv @ dv)


def transelate(particles, n, dt):
    particles[n, :, :2] = particles[n, :, :2] + particles[n, :, 2:] * dt


def collide(particles, n, i, j,  collision_type, radii, masses, xi, xi_p):
    if collision_type == "wall0":
        particles[n, i, 2:] = xi * np.array([-particles[n, i, 2], particles[n, i, 3]])
    if collision_type == "wall1":
        particles[n, i, 2:] = xi * np.array([particles[n, i, 2], -particles[n, i, 3]])
    elif collision_type == "particle":
        R = radii[i] + radii[j]
        dx = particles[n, j, :2] - particles[n, i, :2]
        dv = particles[n, j, 2:] - particles[n, i, 2:]
        mu = masses[i] * masses[j] / (masses[i] + masses[j])
        a = (1 + xi_p) * mu * (dv@dx)/R**2
        particles[n, i, 2:] += a / masses[i] * dx
        particles[n, j, 2:] += -a / masses[j] * dx


"""
Main Loop
"""

def run_loop(init, N, T, radii, masses, xi, xi_p):
    tic = time.time()
    print("Placing particles")
    particles = np.empty((T+1, N, 4))
    particles[0] = init(N, radii)
    print("Finding inital collisions")
    collisions = init_collisions(particles, radii)
    # When has particle i last collided? Used to remove invalid collisions
    last_collided = -np.ones(N)

    t = np.zeros(T+1)
    n = 0
    bar = Bar("running simulation", max=T)
    while n < T:
        t_next, i, j, t_added, col_type = get_next_col(collisions)
        
        # Skip invalid collisions
        valid_collision = (t_added >= last_collided[i]) \
            and (j==-1 or (t_added >= last_collided[j]))

        if valid_collision:
            particles[n+1] = particles[n]
            dt = t_next - t[n]
            t[n+1] = t_next

            transelate(particles, n+1, dt)
            collide(particles, n+1, i, j, col_type, radii, masses, xi, xi_p)

            last_collided[i] = t[n+1]
            push_next_collision(particles, n+1, i, t[n+1], collisions, radii)
            if j !=-1: 
                last_collided[j] = t[n+1]
                push_next_collision(particles, n+1, j, t[n+1], collisions, radii)

            n += 1
            t[n] += 1e-9
            bar.next()
        
    bar.finish()
    print("Time elapsed: {}".format(time.time() - tic))
    return particles, t
