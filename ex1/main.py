import numpy as np
import sys

from utillities import check_crater_size, read_data, simulate, run_loop, energy_condition
from particle_init import *
from plotting import *


# Path for saving data
data_dir = "data/"
# Path for reading parameters
para_dir = "parameters/"
plot_dir = "plots/"

def test_case_one_particle():
    name = "test_case_one_particle"
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
    anim_particles(particles, t, N, radii, 1, title=name)
    plot_energy(particles, t, masses, plot_dir + name + "/")


def test_case_two_particles():
    name = "test_case_two_particles"
    xi = 0.8
    N = 2
    T = 200
    R = 0.1
    radii = np.ones(N) * R
    masses = np.ones(N)
    args = (N, T, radii, masses, xi)

    particles, t = run_loop(init_two_testparticles, args)
    anim_particles(particles, t, N, radii, 5, title=name)
    plot_energy(particles, t, masses, plot_dir + name + "/")


def test_case_many_particles():
    name = "test_case_many_particles"
    xi = 1
    N = 100
    T = 1000
    R = 0.02
    radii = np.ones(N) * R
    masses = np.ones(N)
    args = (N, T, radii, masses, xi)

    particles, t = run_loop(random_dist, args)
    anim_particles(particles, t, N, radii, 0.03, intr=150, title=name)
    plot_energy(particles, t, masses, plot_dir + name + "/")


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
    name = "problem1"
    path = data_dir + name + "/" 
    xi = 1
    N = 200
    T = 1000
    R = 0.01
    radii = np.ones(N) * R
    masses = np.ones(N)

    args = (N, T, radii, masses, xi)
    if run_simulation: simulate(path, random_dist, args)
    
    else:
        particles, t = read_data(path)
        plot_vel_dist(particles, 5*N, N, masses, plot_dir + name + "/")


def problem2(run_simulation=False):
    name = "problem2"
    path = data_dir + name + "/" 
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
        plot_vel_dist(particles[:, :N1], 5*N, N, masses[:N1], plot_dir + name + "/m=1/")
        plot_vel_dist(particles[:, N1:], 5*N, N, masses[N1:], plot_dir + name + "/m=4/")


def problem3(run_simulation=False):
    name = "problem3"
    path = data_dir + name + "/" 
    xis = [1, 0.9, 0.8]
    N = 500
    T = 2_000
    R = 0.01
    radii =np.ones(N) * R
    masses = np.empty(N)
    N1 = N//2
    N2 = N - N1
    masses[:N//2] = np.ones(N1)
    masses[N//2:] = 4 * np.ones(N2)

    if run_simulation: 
        for i, xi in enumerate(xis):
            path_xi = path + "xi_" + str(i) + "/"
            args = (N, T, radii, masses, xi)
            simulate(path_xi, random_dist, args, TC=True)

    else:
        for i, xi in enumerate(xis):
            dir = "xi_" + str(i) + "/"
            path_xi = path + dir
            particles, t = read_data(path_xi)
            plot_energy_prob3(particles, t, masses, N1, N2, plot_dir + name + "/" + dir)

    
def test_case_projectile(run_simulation=False):
    name = "test_case_projectile"
    xi = 0.5
    N = 2000 + 1
    T = 100_000
    R = 0.0054

    radii = np.ones(N) * R
    radii[0] = 0.05
    masses = np.ones(N)
    masses[0] = 25
    args = (N, T, radii, masses, xi)

    path = data_dir + name + "/"

    if run_simulation:
        init = lambda N, radii : init_projectile(N, radii, 5)
        simulate(path, init, args)

    else:
        particles, t = read_data(path)
        anim_particles(particles, t, N, radii, 0.005, title=name)


def problem4(vals, run_simulation=False):
    name = "problem4"
    xi = 0.5
    T = 30_000
    N = 2000 + 1
    R = 0.0056

    radii = np.ones(N) * R
    masses = np.ones(N) * R**2

    m = 10
    Rs = np.linspace(0.01, 0.03, m)

    if run_simulation:
        for i, R in enumerate(Rs):
            radii[0] = R
            masses[0] = R**2
            args = (N, T, radii, masses, xi)

            path = data_dir + name + "/sweep_{}/".format(i)
            init = lambda N, radii : init_projectile(N, radii, 20)
            simulate(path, init, args, condition=energy_condition, n_check=100, TC=True)

    else:
        crater_size = np.zeros(m)
        for i, R in enumerate(Rs):
            radii[0] = R
            masses[0] = R**2
            args = (N, T, radii, masses, xi)

            path = data_dir + name + "/sweep_{}/".format(i)
            particles, t = read_data(path)
            dx = 0.015
            y_max = 0.5
            free_space = check_crater_size(particles, radii, -1, y_max, dx)
            crater_size[i] = dx**2 * np.sum(free_space)
            dir_path = "plots/" + name + "/"
            plot_particles(particles, -1, N, radii, dir_path, fname="particles{}".format(i))
            plot_crater(free_space, y_max, dir_path, fname="crater{}".format(i))
        
        plot_crater_size(Rs, crater_size)


tests = [
    test_case_one_particle, 
    test_case_two_particles, 
    test_case_many_particles, 
    test_case_collision_angle,
    test_case_projectile
    ]

problems = [
    problem1,
    problem2,
    problem3,
]

def cl_arguments(args):
    """ function for processing arguments from the command line """
    
    # run the tests. syntax: python main.py test 0 2 (run) for test 0 and 2 (run simulation)
    if args[1] == "test":
        for arg in args[2:]:
            if args[-1] == "run":
                tests[int(arg)](True)
            else:
                tests[int(arg)]()

    elif args[1] == "problem":
        for arg in args[2:]:
            try: 
                int(arg)
            except:
                break
            if args[-1] == "run":
                problems[int(arg)](True)
            else:
                problems[int(arg)]()

    elif args[1] == "sweep":
        if args[2] == "run":
            vals = args[2:]
            problem4(vals, True)


if __name__ == "__main__":
    cl_arguments(sys.argv)
    # profile_run()

    # problem1(True)
    # problem1()
    # problem2(True)
    # problem2()
    # problem3()

    # test_case_projectile(True)
    # test_case_projectile()

    # problem4(True)
    # problem4()

