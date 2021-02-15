import numpy as np

from utillities import cheack_crater_size, read_data, simulate, run_loop, energy_condition
from particle_init import *
from plotting import *


# Path for saving data
data_folder = "data/"

def test_case_one_particle():
    # Elasticity parametre
    xi = 1
    # Number of particles
    N = 1
    # Number of timesteps
    T = 100
    # Radius
    R = 0.1
    radii = np.ones(N) * R
    masses = np.ones(N)
    args = (N, T, radii, masses, xi)

    particles, t = run_loop(init_one_testparticle, args)
    anim_particles(particles, t, N, radii, 1, title="test_case_one_particle")
    plot_energy(particles, t, masses)


def test_case_two_particles():
    xi = 0.8
    N = 2
    T = 200
    R = 0.1
    radii = np.ones(N) * R
    masses = np.ones(N)
    args = (N, T, radii, masses, xi)

    particles, t = run_loop(init_two_testparticles, args)
    anim_particles(particles, t, N, radii, 5, title="test_case_two_particles")
    plot_energy(particles, t, masses)


def test_case_many_particles():
    xi = 1
    N = 100
    T = 1000
    R = 0.02
    radii = np.ones(N) * R
    masses = np.ones(N)
    args = (N, T, radii, masses, xi)

    particles, t = run_loop(random_dist, args)
    anim_particles(particles, t, N, radii, 0.03, intr=150, title="test_case_man_particles")
    plot_energy(particles, t, masses)


def test_case_collision_angle():
    xi = 1
    N = T = 2
    a = 0.01
    R = 1e-6
    radii = np.array([a, R])
    masses = np.array([1e6, 1])
    args = (N, T, radii, masses, xi)

    m = 100
    bs = np.linspace(-a , a, m)
    theta = np.empty(m)
    for i, b in enumerate(bs):
        init = lambda N, radii : init_collision_angle(b, N, radii)
        particles, t = run_loop(init, args)
        x, y = particles[2, 1, :2]
        x -= 0.5
        y -= 0.5
        theta[i] = np.arctan2(y, -x)
    plot_collision_angle(theta, bs, a)    


def profile_run():
    # https://web.archive.org/web/20140513005858im_/http://www.appneta.com/blog/line-profiler-python/
    xi = 1
    N = 500
    T = 1000
    R = 0.002
    radii = np.ones(N) * R
    masses = np.ones(N)
    args = (N, T, radii, masses, xi)
    particles, t = run_loop(random_dist, args)


def problem1(run_simulation = False):
    path = data_folder + "problem1/"
    xi = 1
    N = 2_000
    T = 50_000
    R = 0.002
    radii = np.ones(N) * R
    masses = np.ones(N)

    args = (N, T, radii, masses, xi)
    if run_simulation: simulate(path, random_dist, args)
    
    else:
        particles, t = read_data(path)
        plot_vel_dist(particles, 5*N, N, masses)


def problem2(run_simulation=False):
    path = data_folder + "problem2/"
    xi = 1
    N = 200
    T = 2_000
    R = 0.01
    radii = np.ones(N) * R
    masses = np.empty(N)
    N1 = N//2
    N2 = N - N1
    masses[:N//2] = np.ones(N1)
    masses[N//2:] = 4 * np.ones(N2)

    args = (N, T, radii, masses, xi)
    if run_simulation: simulate(path, random_dist, args)

    else:
        particles, t = read_data(path)
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
            args = (N, T, radii, masses, xi)
            simulate(path_xi, random_dist, args)

    else:
        for i, xi in enumerate(xis):
            path_xi = path + str(i) + "/"
            particles, t = read_data(path_xi)
            plot_energy_prob3(particles, t, masses, N1, N2)


def projectile(v_proj, path, args):
    init = lambda N, radii : init_projectile(N, radii, v_proj)
    simulate(path, init, args, condition=energy_condition, n_check=100, TC=True)

    
def single_projectile(run_simulation=False):
    xi = 0.5
    N = 2000 + 1
    T = 100_000
    R = 0.0054
    # Test values

    radii = np.ones(N) * R
    radii[0] = 0.05
    masses = np.ones(N)
    masses[0] = 25
    args = (N, T, radii, masses, xi)

    path = data_folder + "single_part/"

    if run_simulation:
        init = lambda N, radii : init_projectile(N, radii, 5)
        simulate(path, init, args)

    else:
        particles, t = read_data(path)
        anim_particles(particles, t, N, radii, 0.005, title="projectile")


def parametre_sweep(run_simulation=False):
    xi = 0.5
    N = 10 + 1
    T = 10_0
    R = 0.008
    radii = np.ones(N) * R
    radii[0] = 0.05
    masses = np.ones(N)
    masses[0] = 25
    args = (N, T, radii, masses, xi)

    m = 10
    vs = np.linspace(1, 20, m)

    if run_simulation:
        for i, v in enumerate(vs):
            path = data_folder + "problem4/sweep_{}/".format(i)
            projectile(v, path, args)

    else:
        crater_size = np.zeros(m)
        for i, v in enumerate(vs):
            path = data_folder + "problem4/sweep_{}/".format(i)
            particles, t = read_data(path)
            dx = 0.04
            y_max = 0.5
            free_space = cheack_crater_size(particles, -1, y_max, dx)
            crater_size[i] = dx**2 * np.sum(free_space)
            plot_particles(particles, -1, N, radii)
            plot_crater(free_space, y_max, dx)
        plot_crater_size(vs, crater_size)

if __name__ == "__main__":
    # test_case_one_particle()
    # test_case_two_particles()
    # test_case_many_particles()
    # test_case_collision_angle()

    # profile_run()

    # problem1(True)
    # problem1()
    # problem2(True)
    # problem2()
    # problem3()

    # single_projectile(True)
    # single_projectile()

    parametre_sweep(True)
    parametre_sweep()