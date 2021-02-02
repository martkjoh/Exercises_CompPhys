import heapq
import os
import numpy as np
from progress.bar import Bar
import matplotlib.pyplot as plt # to be removed

from utillities import init_collisions, transelate, collide, push_next_collision
from particle_init import *
from plotting import anim_particles, plot_particles, plot_vel_dist, plot_energy

# Path for saving data
data_folder = "data/"


# Main loop

def run_loop(init, N, T, radii, masses, xi, xi_p):
    particles = np.empty((T+1, N, 4))
    particles[0] = init(N, radii)
    collisions = init_collisions(particles, radii)
    # When has particle i last collided? Used to remove invalid collisions
    last_collided = -np.ones(N)

    t = np.zeros(T+1)
    n = 0
    bar = Bar("running simulation", max=T)
    while n < T:
        next_coll = heapq.heappop(collisions)
        t_next, i, j, t_added, collision_type = next_coll
        
        # Skip invalid collisions
        valid_collision = (t_added >= last_collided[i]) \
            and (j==-1 or (t_added >= last_collided[j]))

        if valid_collision:
            particles[n+1] = particles[n]
            dt = t_next - t[n]
            t[n+1] = t_next

            transelate(particles, n+1, dt)
            collide(particles, n+1, i, j, collision_type, radii, masses, xi, xi_p)

            last_collided[i] = t[n+1]
            push_next_collision(particles, n+1, i, t[n+1], collisions, radii)
            if j !=-1: 
                last_collided[j] = t[n+1]
                push_next_collision(particles, n+1, j, t[n+1], collisions, radii)

            n += 1
            bar.next()
        
    bar.finish()
    return particles, t


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


def profile_run():
    # uncomment the @profile decorator, and run kernprof.py -l -v example.py
    # https://web.archive.org/web/20140513005858im_/http://www.appneta.com/blog/line-profiler-python/
    xi = 1
    xi_p = 1
    N = 500
    T = 1000
    R = 0.002
    radii = np.ones(N) * R
    masses = np.ones(N)
    particles, t = run_loop(random_dist, N, T, radii, masses, xi, xi_p)


def problem1(run_simulation = False):
    path = data_folder + "problem1/"
    xi = 1
    xi_p = 1
    N = 1000
    T = 10_000
    R = 0.002
    radii = np.ones(N) * R
    masses = np.ones(N)

    if run_simulation:
        particles, t = run_loop(random_dist, N, T, radii, masses, xi, xi_p)
        np.save(path + "particles.npy", particles)
        np.save(path + "t.npy", t)
    
    particles = np.load(path + "particles.npy")
    t = np.load(path + "t.npy")

    plot_vel_dist(particles, 3000, 1000, masses)




if __name__ == "__main__":
    # profile_run()
    # problem1()
    test_case_many_particles()
    test_case_collision_angle()