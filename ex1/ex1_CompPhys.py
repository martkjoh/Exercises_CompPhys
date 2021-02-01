from matplotlib import collections
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.animation import FuncAnimation as FA
import heapq


"""
Initializing
"""

# Sides of the box
L = 1

def init_one_testparticle(N, radii):
    R = radii[0]
    return np.array([[R, 1/2, 1, -1],])


def init_two_testparticles(N, radii):
    R = radii[0]
    return np.array([
        [R, 0.5, 0, 0],
        [0.5, 1-R, 0, 0]])

def init_collision_angle(b, N, radii):
    return np.array([
        [1/2, 1/2, 0, 0],
        [radii[1], 1/2 + b, 1, 0]
    ])

# Particles must wholly inside the box, and not overlapping
def random_dist(N, radii):
    # particle_no, (x, y, vx, vy)
    particles = np.zeros((N, 4))
    i = 0
    k = 0
    while i<N:
        pos = np.random.rand(2)

        # Check if inside box
        if (pos[0] - radii[i]) < 0 or (pos[0] + radii[i]) > L: continue
        if (pos[1] - radii[i]) < 0 or (pos[1] + radii[i]) > L: continue

        # Check if overlap with other particles
        overlap = False
        for j in range(i):
            dist = (pos[0] - particles[j, 0])**2 + (pos[1] - particles[j, 1])**2
            if  dist < (radii[i] + radii[j])**2:
                overlap = True
                break
        
        if not overlap:
            vel = np.random.rand(2) - 0.5
            vel = vel/np.sqrt(vel[0]**2 + vel[1]**2)
            particles[i] = np.array([pos[0], pos[1], vel[0], vel[1]])
            i+=1

        # emergency break (heh)
        else: k += 1
        if k > N*10: 
            raise Exception("can't fit particles")
    
    return particles


"""
Utillities
"""

def get_vel2(particles, n):
    return np.einsum("ij -> i", particles[n, :, 2:]**2)


def get_energy(particles, masses, n):
    return 1/2 * masses @ (get_vel2(particles, n))


def get_temp(particles, masses, n):
    N = len(particles[n])
    return get_energy(particles, masses, n) / N


def MaxBoltz(v, m, T):
    return m * v / T * np.exp(-m * v**2 / (2 * T))


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


def push_next_collision(particles, n, i, t, collisions, radii):
    wall0 = check_wall_collison(particles[n, i, 0], particles[n, i, 2], radii[i])
    wall1 = check_wall_collison(particles[n, i, 1], particles[n, i, 3], radii[i])
    heapq.heappush(collisions, (t+wall0, i, -1, t, "wall0"))
    heapq.heappush(collisions, (t+wall1, i, -1, t, "wall1"))
    for j in range(len(particles[0])):
        dt = check_particle_collision(particles, n, i, j, radii)
        heapq.heappush(collisions, (t+dt, i, j, t, "particle"))


def init_collisions(particles, radii):
    collisions = []
    for i in range(len(particles[0])):
        push_next_collision(particles, 0, i, 0, collisions, radii)
    return collisions


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
main loop
"""

def run_loop(init, N, T, radii, masses, xi, xi_p):
    particles = np.empty((T+1, N, 4))
    particles[0] = init(N, radii)
    collisions = init_collisions(particles, radii)
    # When has particle i last collided? Used to remove invalid collisions
    last_collided = -np.ones(N)

    t = np.zeros(T+1)
    n = 0
    while n < T:
        next_coll = heapq.heappop(collisions)
        t_next = next_coll[0]
        i = next_coll[1]
        j = next_coll[2]
        t_added = next_coll[3]
        
        # Skip invalid collisions
        valid_collision = (t_added >= last_collided[i]) \
            and (j==-1 or (t_added >= last_collided[j]))

        if valid_collision:
            particles[n+1] = particles[n]
            dt = t_next - t[n]
            t[n+1] = t_next
            transelate(particles, n+1, dt)
            collide(particles, n+1, i, j, next_coll[4], radii, masses, xi, xi_p)
            last_collided[i] = t[n+1]
            push_next_collision(particles, n+1, i, t[n+1], collisions, radii)
            if j !=-1: 
                last_collided[j] = t[n+1]
                push_next_collision(particles, n+1, j, t[n+1], collisions, radii)
            n += 1
    
    return particles, t


"""
Plotting
"""

def plot_energy(particles, t, masses):
    fig, ax = plt.subplots()
    N = len(t)
    E = np.array([get_energy(particles, masses, n) for n in range(N)])
    T = len(particles)
    ax.plot(np.arange(T), E)
    plt.show()

def plot_vel_dist(particles, n, masses):
    fig, ax = plt.subplots()
    N = len(particles)
    v2 = get_vel2(particles, n)
    Temp = get_temp(particles, masses, n, N)
    ax.hist(np.sqrt(v2), bins=30, density=True)
    v = np.linspace(np.sqrt(np.min(v2)), np.sqrt(np.max(v2)), 1000)
    ax.plot(v, MaxBoltz(v, masses[0], Temp))
    plt.show()


def get_particles_plot(particles, n, N, radii):
    circles =  [plt.Circle(
        (particles[n, i, 0], particles[n, i, 1]),radius=radii[i], linewidth=0) 
        for i in range(N)]
    return circles

def get_arrows_plot(particles, n, N, radii):
    arrows = [plt.Arrow(
        particles[n, i, 0], 
        particles[n, i, 1], 
        particles[n, i, 2]*radii[i], 
        particles[n, i, 3]*radii[i],
        width=radii[i]*0.4)
        for i in range(N)]
    return arrows


def plot_particles(particles, n, N, radii, plot_vel=True):
    fig, ax = plt.subplots()
    ax.set_ylim(0, 1)
    ax.set_xlim(0, 1)
    circles  = get_particles_plot(particles, n, N, radii)
    arrows = get_arrows_plot(particles, n, N, radii)
    patches = PatchCollection(circles + arrows)
    colors = np.concatenate([np.linspace(0.2, 0.8, N), np.zeros(N)])
    patches.set_array(colors)
    ax.set_title(n)

    ax.add_collection(patches)

    plt.show()


def anim_particles(particles, t, N, radii, plot_vel=True):
    dt = 0.1
    steps = np.nonzero(np.diff(t // dt))[0]
    fig, ax = plt.subplots()
    ax.set_ylim(0, 1)
    ax.set_xlim(0, 1)

    circles = get_particles_plot(particles, 0, N, radii)
    arrows = get_arrows_plot(particles, 0, N, radii)

    patches = PatchCollection(circles + arrows)
    colors = np.concatenate([np.linspace(0.2, 0.8, N), np.zeros(N)])
    patches.set_array(colors)

    ax.add_collection(patches)

    def anim(n, fargs):
        steps = fargs
        circles = get_particles_plot(particles, n, N, radii)
        arrows = get_arrows_plot(particles, n, N, radii)
        patches.set_paths(circles + arrows)
        

    a = FA(fig, anim, fargs=(steps,), interval=100)
    plt.show()
    

"""
Running
"""

def test_case_one_particle():
    # Elasticity parametre
    xi = 1
    xi_p = 1
    # Number of particles
    N = 1
    # Number of timesteps
    T = 1000
    # Radius
    R = 0.1
    radii = np.ones(N) * R
    masses = np.ones(N)

    particles, t = run_loop(init_one_testparticle, N, T, radii, masses, xi, xi_p)
    anim_particles(particles, t, N, radii)
    plot_energy(particles, t, masses)

def test_case_two_particles():
    xi = 1
    xi_p = 1
    N = 2
    T = 1000
    R = 0.05
    radii = np.ones(N) * R
    masses = np.ones(N)

    particles, t = run_loop(init_two_testparticles, N, T, radii, masses, xi, xi_p)
    anim_particles(particles, t, N, radii)
    plot_energy(particles, t, masses)

def test_case_many_particles():
    xi = 1
    xi_p = 1
    N = 100
    T = 1000
    R = 0.02
    radii = np.ones(N) * R
    masses = np.ones(N)

    particles, t = run_loop(random_dist, N, T, radii, masses, xi, xi_p)
    anim_particles(particles, t, N, radii)
    plot_energy(particles, t, masses)

def test_case_collision_angle():
    xi = 1
    xi_p = 1
    N = 2
    T = 2
    a = 0.01
    R = 1e-6
    radii = np.array([a, R])
    masses = np.array([1e6, 1])
    m = 100
    bs = np.linspace(-a , a, m)
    theta = np.empty(m)
    for i, b in enumerate(bs):
        init = lambda N, radii : init_collision_angle(b, N, radii)
        particles, t = run_loop(init, N, T, radii, masses, xi, xi_p)
        x, y = particles[2, 1, :2]
        x -= 0.5
        y -= 0.5
        theta[i] = np.arctan2(y, -x)
    fig, ax = plt.subplots()
    ax.plot(theta, bs)
    ax.plot(theta, a *  np.sin(theta / 2), "k--")
    plt.show()


# for i in range(10):
#     plot_vel_dist(particles, int(T/10 * i)+1)


test_case_collision_angle()