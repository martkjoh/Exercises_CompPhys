import heapq
import numpy as np


# Sides of the box
L = 1 

"""
utillities
"""


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
