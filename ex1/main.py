import numpy as np
import matplotlib.pyplot as plt # to be removed

from utillities import init_collisions, read_data, transelate, collide, push_next_collision, get_next_col, simulate, run_loop
from particle_init import *
from plotting import anim_particles, plot_particles, plot_vel_dist, plot_energy, plot_energy_prob3


# Path for saving data
data_folder = "data/"

def test_case_one_particle():
    # Elasticity parametre
    xi = xi_p = 1
    # Number of particles
    N = 1
    # Number of timesteps
    T = 1000
    # Radius
    R = 0.1
    radii = np.ones(N) * R
    masses = np.ones(N)

    particles, t = run_loop(init_one_testparticle, N, T, radii, masses, xi, xi_p)
    anim_particles(particles, t, N, radii, title="test_case_one_particle")
    plot_energy(particles, t, masses)


def test_case_two_particles():
    xi = xi_p = 1
    N = 2
    T = 1000
    R = 0.05
    radii = np.ones(N) * R
    masses = np.ones(N)

    particles, t = run_loop(init_two_testparticles, N, T, radii, masses, xi, xi_p)
    anim_particles(particles, t, N, radii, title="test_case_two_particles")
    plot_energy(particles, t, masses)


def test_case_many_particles():
    xi = xi_p = 1
    N = 100
    T = 100
    R = 0.02
    radii = np.ones(N) * R
    masses = np.ones(N)

    particles, t = run_loop(random_dist, N, T, radii, masses, xi, xi_p)
    anim_particles(particles, t, N, radii, "test_case_man_particles")
    plot_energy(particles, t, masses)


def test_case_collision_angle():
    xi = xi_p = 1
    N = T = 2
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
    # https://web.archive.org/web/20140513005858im_/http://www.appneta.com/blog/line-profiler-python/
    xi = xi_p = 1
    N = 500
    T = 1000
    R = 0.002
    radii = np.ones(N) * R
    masses = np.ones(N)
    particles, t = run_loop(random_dist, N, T, radii, masses, xi, xi_p)


def problem1(run_simulation = False):
    path = data_folder + "problem1/"
    xi = xi_p = 1
    N = 2_000
    T = 50_000
    R = 0.002
    radii = np.ones(N) * R
    masses = np.ones(N)

    kwargs = (random_dist, N, T, radii, masses, xi, xi_p)
    if run_simulation: simulate(path, kwargs)
    
    else:
        particles, t = read_data(path)
        plot_vel_dist(particles, 5*N, N, masses)


def problem2(run_simulation=False):
    path = data_folder + "problem2/"
    xi = xi_p = 1
    N = 200
    T = 2_000
    R = 0.01
    radii = np.ones(N) * R
    masses = np.empty(N)
    N1 = N//2
    N2 = N - N1
    masses[:N//2] = np.ones(N1)
    masses[N//2:] = 4 * np.ones(N2)

    kwargs = (random_dist, N, T, radii, masses, xi, xi_p)
    if run_simulation: simulate(path, kwargs)

    else:
        particles, t = read_data(path)
        # anim_particles(particles, t, N, radii)
        plot_vel_dist(particles[:, :N1], 5*N, N, masses[:N1])
        plot_vel_dist(particles[:, N1:], 5*N, N, masses[N1:])


def problem3(run_simulation=False):
    path = data_folder + "problem3/"
    xis = [1, 0.9, 0.8]
    N = 1_500
    T = 20_000
    R = 0.004
    radii =np.ones(N) * R
    masses = np.empty(N)
    N1 = N//2
    N2 = N - N1
    masses[:N//2] = np.ones(N1)
    masses[N//2:] = 4 * np.ones(N2)

    if run_simulation: 
        for i, xi in enumerate(xis):
            path_xi = path + str(i) + "/"
            kwargs = (random_dist, N, T, radii, masses, xi, xi)
            simulate(path_xi, kwargs)

    else:
        for i, xi in enumerate(xis):
            path_xi = path + str(i) + "/"
            particles, t = read_data(path_xi)
            plot_energy_prob3(particles, t, masses, N1, N2)


def problem4(run_simulation=False):
    path = data_folder + "problem4/"
    xi = xi_p = 0.5
    N = 1000 + 1
    T = 20_000
    R = 0.0075
    radii = np.ones(N) * R
    radii[0] = 0.05
    masses = np.ones(N)
    masses[0] = 25

    kwargs = (projectile, N, T, radii, masses, xi, xi_p)
    if run_simulation: simulate(path, kwargs)

    else:
        particles, t = read_data(path)
        anim_particles(particles, t, N, radii, title="projectile")


if __name__ == "__main__":
    # test_case_many_particles()
    # test_case_collision_angle()

    # profile_run()

    # problem1()
    # problem2(True)
    # problem2()
    # problem3()
    problem4(True)
    problem4()
