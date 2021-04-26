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


def push_wall_collision(particles, i, n, t, collisions, radii):
    wall0 = check_wall_collison(particles[n, i, 0], particles[n, i, 2], radii[i])
    wall1 = check_wall_collison(particles[n, i, 1], particles[n, i, 3], radii[i])
    if wall0 != np.inf: heapq.heappush(collisions, (t+wall0, i, -1, t, "wall0"))
    if wall1 != np.inf: heapq.heappush(collisions, (t+wall1, i, -1, t, "wall1"))


def push_particle_collision(particles, n, i, t, collisions, radii):
    dts, cond = check_particle_collisions(particles, n, i, radii)
    for j, a in enumerate(cond):
        if a:
            j_true = j
            if j >= i: j_true += 1
            heapq.heappush(collisions, (t+dts[j], i, j_true, t, "particle"))


def push_next_collision(particles, n, i, t, collisions, radii):
    push_wall_collision(particles, i, n, t, collisions, radii)
    push_particle_collision(particles, n, i, t, collisions, radii)


def init_collisions(particles, radii, t=0):
    collisions = []
    N = len(particles[0])
    for i in range(N):
        push_next_collision(particles, 0, i, t, collisions, radii)
    return collisions


def make_dir(dir_path):
    """ recursively (!) creates the needed directories """
    if not path.isdir(dir_path):
        make_dir("/".join(dir_path.split("/")[:-2]) + "/")
        mkdir(dir_path)


def check_dir(dir_path):
    if not path.isdir(dir_path):
        make_dir(dir_path)


def save_data(particles, t, dir_path, N_save):
    check_dir(dir_path)
    np.save(dir_path + "particles.npy", particles)
    np.save(dir_path + "t.npy", t)


def read_data(path):
    print("Reading particle data from " + path)
    particles = np.load(path + "particles.npy")
    t = np.load(path + "t.npy")
    return particles, t


def read_params(path):
    params = np.loadtxt(path + ".txt")
    xi, N, T, R, N_save = params[:5]
    N, T, N_save = int(N), int(T), int(N_save)
    if len(params) > 5:
        return xi, N, T, R, N_save, params[5:]
    else:
        return xi, N, T, R, N_save


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


def check_particle_collisions(particles, n, i, radii):
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


def energy_condition(system, args, n):
    N, T, radii, masses, xi, N_save = args
    t, particles, collisions, last_collided = system
    E0 = get_energy(particles, masses, 0)
    E = get_energy(particles, masses, n)
    ratio = E/E0
    print("\nenergy ratio: ", ratio, end="")
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


def check_crater_size(particles, radii, n, Nx):
    print("Measuring crater")
    ymax = 0.5
    tic = time.time()
    Ny = int(Nx*ymax)
    x, dx= np.linspace(0, 1, Nx, retstep=True, endpoint=False)
    y = np.linspace(0, ymax, Ny)
    indices_x = np.arange(Nx)
    indices_y = np.arange(Ny)
    free_space = np.zeros((Nx, Ny))

    for i in range(len(particles[n])):
        x0, y0 = particles[n, i, 0:2]
        R = radii[i]
        # What coordinates might the particles be inside
        x_might = np.logical_and(x>(x0-R-dx), x<(x0+R+2*dx))
        y_might = np.logical_and(y>(y0-R-dx), y<(y0+R+2*dx))
        for j in indices_x[x_might]:
            for k in indices_y[y_might]:
                # Check if actually is inside
                is_inside = particle_insde(j*dx, k*dx, dx, radii[i], particles[n, i])
                if is_inside:
                    free_space[j, k] = 1.

    print("Time measuring crater: {}".format(time.time() - tic))
    return free_space


"""
Main Loop
"""


def setup_loop(init, args):
    N, T, radii, masses, xi, N_save = args
    print("Placing particles")
    particles = np.empty((N_save, N, 4))
    tic = time.time()
    particles[0] = init(N, radii)
    print("Time placing particles: {}".format(time.time() - tic))
    print("Finding inital collisions")
    collisions = init_collisions(particles, radii)
    # When has particle i last collided? Used to remove invalid collisions
    last_collided = -np.ones(N)
    t = np.zeros(N_save)

    return t, particles, collisions, last_collided


def execute_collision(k, kp1, system, args, col, TC):
    # This got out of hand
    N, T, radii, masses, xi, N_save = args
    t, particles, collisions, last_collided = system
    t_next, i, j, t_added, col_type = col

    particles[kp1] = particles[k]
    dt = col[0] - t[k]
    t[kp1] = t_next

    if TC:
        xi = tc_check(i, k, t, last_collided, xi)
        if j!=-1: xi = tc_check(j, k, t, last_collided, xi)

    transelate(particles, kp1, dt)
    collide(particles, kp1, i, j, col_type, radii, masses, xi)
    last_collided[i] = t[kp1]
    push_next_collision(particles, kp1, i, t[kp1], collisions, radii)
    if j !=-1: 
        last_collided[j] = t[kp1]
        push_next_collision(particles, kp1, j, t[kp1], collisions, radii)

    return t, particles, collisions, last_collided


def run_check(system, check, args, kp1):
    """Checks if the loop has slowed down significantly, can delete cumulated collisions"""
    t, particles, collisions, last_collided = system
    N, T, radii, masses, xi, N_save = args
    check.append((time.time(), len(collisions)))
    dt = (check[-1][0]-check[-2][0])
    dt0 = (check[1][0]-check[0][0])
    if dt/dt0>4 and check[-1][1]>check[-2][1]:
        print("\ndiscarding collisions")
        collisions = init_collisions(particles[kp1:], radii, t=t[kp1])
        system = t, particles, collisions, -np.ones(N)

    print("\ntime since last check: {}, number of collisions: {}".format(dt, check[-1][1]))
    return system, check


def run_loop(init, args, condition=None, TC=False):
    tic = time.time()
    system = setup_loop(init, args) # tuple describing the simulated system
    t, particles, collisions, last_collided = system
    N, T, radii, masses, xi, N_save = args
    assert (T-1)%(N_save-1)==0
    skip = (T-1)//(N_save-1)
    
    n = 0 # index for how many collisions have happened
    k = 0 # Index for reading out of particles
    kp1 = 1 # Index for writing to particles, =k or =k+1
    bar = Bar("running simulation", max=T-1)
    check = [(time.time(), len(collisions))]

    while n < T-1:
        t, particles, collisions, last_collided = system
        col = heapq.heappop(collisions)
        t_next, i, j, t_added, col_type = col
        
        # skip invalid collisions
        valid_collision = (t_added >= last_collided[i]) \
            and (j==-1 or (t_added >= last_collided[j]))

        if valid_collision:
            system = execute_collision(k, kp1, system, args, col, TC)
            
            if (kp1-k)>0 and k!=0:
                if not(condition is None) and condition(system, args, kp1): break
                system, check = run_check(system, check, args, kp1)

            n += 1
            k = (skip+n-1) // skip
            kp1 = (skip+n) // skip
            bar.next()

    bar.finish()
    print("Time elapsed: {}".format(time.time() - tic))
    t, particles, collisions, last_collided = system
    return particles[:kp1], t[:kp1]
