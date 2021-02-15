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


def make_dir(dir_path):
    """ recursively (!) creates the needed directories """
    if not path.isdir(dir_path):
        make_dir("/".join(dir_path.split("/")[:-2]) + "/")
        mkdir(dir_path)


def check_dir(dir_path):
    if not path.isdir(dir_path):
        make_dir(dir_path)


def simulate(dir_path, init, args, condition=None, n_check=np.inf, TC=False):
    particles, t = run_loop(init, args, condition, n_check, TC=TC)
    check_dir(dir_path)
    np.save(dir_path + "particles.npy", particles)
    np.save(dir_path + "t.npy", t)


def read_data(path):
    print("Reading particle data from " + path)
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


def collide(particles, n, i, j,  collision_type, radii, masses, xi):
    if collision_type == "wall0":
        particles[n, i, 2:] = xi * np.array([-particles[n, i, 2], particles[n, i, 3]])
    if collision_type == "wall1":
        particles[n, i, 2:] = xi * np.array([particles[n, i, 2], -particles[n, i, 3]])
    elif collision_type == "particle":
        R = radii[i] + radii[j]
        dx = particles[n, j, :2] - particles[n, i, :2]
        dv = particles[n, j, 2:] - particles[n, i, 2:]
        mu = masses[i] * masses[j] / (masses[i] + masses[j])
        a = (1 + xi) * mu * (dv@dx)/R**2
        particles[n, i, 2:] += a / masses[i] * dx
        particles[n, j, 2:] += -a / masses[j] * dx


def energy_condition(particles, args, n):
    N, T, radii, masses, xi = args
    E0 = get_energy(particles, masses, 0)
    E = get_energy(particles, masses, n)
    ratio = E/E0
    print(" ", ratio)
    return ratio<0.1


def tc_check(i, n, t, last_collided, xi):
    tc = 1e-8
    dt = t[n] - last_collided[i]
    if dt < tc: return 1
    else: return xi


def particle_insde(x, y, dx, R, particle):
    x0, y0 = particle[0], particle[1]
    # Is the center of the particle inside an area around the square?
    center_inside =  \
        (x - R) < x0 and (x + dx + R) > x0 and\
        (y - R) < y0 and (y + dx + R) > y0
    # Is the center outside the square, at the corners?
    centre_at_corners = \
        (x > x0 and y > y0) or \
        (x > x0 and y + dx < y0) or \
        (x + dx < x0 and y > y0) or \
        (x + dx < x0 and y + dx < y0)
    corners = [(x, y), (x + dx, y), (x, y + dx), (x + dx, y + dx)]
    # Does the particle overlap with a disk, centered at the corners?
    overlap_corner = False
    for corner in corners:
        dist = (corner[0] - x0)**2 + (corner[1] - y0)**2
        overlap_corner = overlap_corner or (dist < (R)**2)
    
    return center_inside
    
    return (center_inside and not centre_at_corners) or overlap_corner


def check_crater_size(particles, radii, n, y_max, dx):
    Nx = int(1 / dx)
    Ny = int(y_max / dx)
    N = len(particles[0])
    free_space = np.zeros((Nx, Ny))
    for i in range(Nx):
        for j in range(Ny):
            is_inside = False
            for k in range(1, N):
                is_inside = particle_insde(i*dx, j*dx, dx, radii[k], particles[n, k])
                if is_inside:
                    free_space[i, j] = 1. 
                    break
    return free_space


"""
Main Loop
"""

def setup_loop(init, args):
    N, T, radii, masses, xi = args
    print("Placing particles")
    particles = np.empty((T+1, N, 4))
    particles[0] = init(N, radii)
    print("Finding inital collisions")
    collisions = init_collisions(particles, radii)
    # When has particle i last collided? Used to remove invalid collisions
    last_collided = -np.ones(N)

    t = np.zeros(T+1)

    return t, particles, collisions, last_collided


def execute_collision(n, t, particles, collisions, last_collided, args, col, TC):
    N, T, radii, masses, xi = args
    t_next, i, j, t_added, col_type  = col
    particles[n+1] = particles[n]
    dt = col[0] - t[n]
    t[n+1] = t_next

    if TC:
        xi = tc_check(i, n, t, last_collided, xi)
        if j!=-1: xi = tc_check(j, n, t, last_collided, xi)

    transelate(particles, n+1, dt)
    collide(particles, n+1, i, j, col_type, radii, masses, xi)
    last_collided[i] = t[n+1]
    push_next_collision(particles, n+1, i, t[n+1], collisions, radii)
    if j !=-1: 
        last_collided[j] = t[n+1]
        push_next_collision(particles, n+1, j, t[n+1], collisions, radii)

    n += 1
    return n, t, particles, collisions, last_collided


def run_loop(init, args, condition=None, n_check=np.inf, TC=False):
    tic = time.time()
    t, particles, collisions, last_collided = setup_loop(init, args)
    N, T, radii, masses, xi = args

    n = 0
    bar = Bar("running simulation", max=T)
    while n < T:
        col = get_next_col(collisions)
        t_next, i, j, t_added, col_type  = col
        
        # Skip invalid collisions
        valid_collision = (t_added >= last_collided[i]) \
            and (j==-1 or (t_added >= last_collided[j]))

        if valid_collision:
            n, t, particles, collisions, last_collided = \
                execute_collision(n, t, particles, collisions, last_collided, args, col, TC)

            if n%n_check==0: 
                if condition(particles, args, n): break

            bar.next()

    bar.finish()
    print("Time elapsed: {}".format(time.time() - tic))
    return particles[:n+1], t[:n+1]
