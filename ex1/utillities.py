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


def push_all_wall_collisions(particles, n, t, colllisions, radii):
    wall0, cond0 = check_wall_collisions_np(particles[n, :, 0], particles[n, :, 2], radii)
    wall1, cond1 = check_wall_collisions_np(particles[n, :, 1], particles[n, :, 3], radii)
    for i in range(len(wall0)):
        if cond0[i]: heapq.heappush(colllisions, (t+wall0[i], i, -1, t, "wall0"))
        if cond1[i]: heapq.heappush(colllisions, (t+wall1[i], i, -1, t, "wall1"))


def push_next_collision_np(particles, n, i, t, collisions, radii):
    wall0 = check_wall_collison(particles[n, i, 0], particles[n, i, 2], radii[i])
    wall1 = check_wall_collison(particles[n, i, 1], particles[n, i, 3], radii[i])
    if wall0 != np.inf: heapq.heappush(collisions, (t+wall0, i, -1, t, "wall0"))
    if wall1 != np.inf: heapq.heappush(collisions, (t+wall1, i, -1, t, "wall1"))
    dts, cond = check_particle_collisions_np(particles, n, i, radii)
    for j, a in enumerate(cond):
        if a:
            j_true = j
            if j >= i: j_true += 1
            heapq.heappush(collisions, (t+dts[j], i, j_true, t, "particle"))


def init_collisions_np(particles, radii):
    collisions = []
    N = len(particles[0])
    push_all_wall_collisions(particles, 0, 0, collisions, radii)
    for i in range(N):
        dts, cond = check_particle_collisions_np(particles, 0, i, radii)
        for j, a in enumerate(cond):
            if a:
                j_true = j
                if j >= i: j_true += 1
                heapq.heappush(collisions, (dts[j], i, j_true, 0, "particle"))
    return collisions


def make_dir(dir_path):
    """ recursively (!) creates the needed directories """
    if not path.isdir(dir_path):
        make_dir("/".join(dir_path.split("/")[:-2]) + "/")
        mkdir(dir_path)


def check_dir(dir_path):
    if not path.isdir(dir_path):
        make_dir(dir_path)


def simulate_np(dir_path, init, args,  condition=None, n_check=np.inf, TC=False):
    return run_loop_np(init, args, condition=None, n_check=np.inf, TC=TC)


def save_data(particles, t, T, dir_path, skip):
    check_dir(dir_path)
    np.save(dir_path + "particles.npy", particles[::skip])
    np.save(dir_path + "t.npy", t[::skip])
    np.save(dir_path + "T.npy", T)


def read_data(path):
    print("Reading particle data from " + path)
    particles = np.load(path + "particles.npy")
    t = np.load(path + "t.npy")
    T = np.load(path + "T.npy")
    return particles, t, T

def read_params(path):
    params = np.loadtxt(path + ".txt")
    xi, N, T, R = params[:4]
    N, T = int(N), int(T)
    if len(params) > 4:
        return xi, N, T, R, params[4:]
    else:
        return xi, N, T, R


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


def check_wall_collisions_np(xs, vs, rs):
    dts = np.full_like(xs, np.inf)
    cond1 = vs>0
    cond2 = vs<0
    dts[cond1] = (L * np.ones_like(xs[cond1]) - rs[cond1] - xs[cond1]) / vs[cond1]
    dts[cond2] = (rs[cond2] - xs[cond2]) / vs[cond2]
    return dts, np.logical_or(cond1, cond2)



def check_particle_collision(particles, n, i, j, radii):
    R = radii[i] + radii[j]
    dx = particles[n, j, :2] - particles[n, i, :2]
    dv = particles[n, j, 2:] - particles[n, i, 2:]
    d = (dv @ dx)**2 - (dv @ dv) * ((dx @ dx) - R**2)
    if (d <= 0 or dv @ dx >= 0): return np.inf
    else: return - (dv @ dx + np.sqrt(d)) / (dv @ dv)


def check_particle_collisions_np(particles, n, i, radii):
    N = len(radii)
    mask = np.arange(N) != i # remove the particle we are checking against
    one = np.ones_like(particles[n, mask, 0])
    one_v = np.array([one, one]).T
    R = radii[i] * one + radii[mask]
    dx = particles[n, mask, :2] - particles[n, i, :2] * one_v
    dv = particles[n, mask, 2:] - particles[n, i, 2:] * one_v
    dvdx = np.einsum("ij -> i", dv*dx)
    dxdx = np.einsum("ij -> i", dx**2)
    dvdv = np.einsum("ij -> i", dv**2)
    d = dvdx**2 - dvdv * (dxdx - R**2)
    cond1 = d <= 0
    cond2 = dvdx >= 0
    cond = np.logical_not(np.logical_or(cond1, cond2))
    dt = one * np.inf
    dt[cond] = - (dvdx[cond] + np.sqrt(d[cond])) / dvdv[cond]
    return dt, cond


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


def setup_loop_np(init, args):
    N, T, radii, masses, xi = args
    print("Placing particles")
    particles = np.empty((T+1, N, 4))
    particles[0] = init(N, radii)
    print("Finding inital collisions")
    collisions = init_collisions_np(particles, radii)
    # When has particle i last collided? Used to remove invalid collisions
    last_collided = -np.ones(N)

    t = np.zeros(T+1)

    return t, particles, collisions, last_collided


def execute_collision_np(n, t, particles, collisions, last_collided, args, col, TC):
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
    push_next_collision_np(particles, n+1, i, t[n+1], collisions, radii)
    if j !=-1: 
        last_collided[j] = t[n+1]
        push_next_collision_np(particles, n+1, j, t[n+1], collisions, radii)

    n += 1
    return n, t, particles, collisions, last_collided


def run_loop_np(init, args, condition=None, n_check=np.inf, TC=False):
    tic = time.time()
    t, particles, collisions, last_collided = setup_loop_np(init, args)
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
                execute_collision_np(n, t, particles, collisions, last_collided, args, col, TC)

            if n%n_check==0: 
                if condition(particles, args, n): break

            bar.next()

    bar.finish()
    print("Time elapsed: {}".format(time.time() - tic))
    return particles[:n+1], t[:n+1]
