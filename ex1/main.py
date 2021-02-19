import numpy as np
import sys

from utillities import check_crater_size, read_data, simulate, run_loop, energy_condition, read_params
from particle_init import *
from plotting import *


# Path for saving data
data_dir = "data/"
# Path for reading parameters
para_dir = "parameters/"
plot_dir = "plots/"

def test_case_one_particle():
    name = "test_case_one_particle"
    xi, N, T, R = read_params(para_dir + name)
    radii = np.ones(N) * R
    masses = np.ones(N)
    args = (N, T, radii, masses, xi)

    particles, t = run_loop(init_one_testparticle, args)

    plot_particles(particles, -2, N, radii, plot_dir + name + "/", "particle-2")
    plot_particles(particles, -1, N, radii, plot_dir + name + "/", "particle-1")
    plot_energy(particles, t, masses, plot_dir + name + "/")


def test_case_two_particles():
    name = "test_case_two_particles"
    xi, N, T, R = read_params(para_dir + name)
    radii = np.ones(N) * R
    masses = np.ones(N)
    args = (N, T, radii, masses, xi)

    particles, t = run_loop(init_two_testparticles, args)
    anim_particles(particles, t, N, radii, 5, title=name)
    plot_energy(particles, t, masses, plot_dir + name + "/")


def test_case_many_particles():
    name = "test_case_many_particles"
    xi, N, T, R = read_params(para_dir + name)
    radii = np.ones(N) * R
    masses = np.ones(N)
    args = (N, T, radii, masses, xi)

    particles, t = run_loop(random_dist, args)
    # anim_particles(particles, t, N, radii, 0.03, intr=150, title=name)
    plot_particles(particles, -1, N, radii, plot_dir + name + "/", name)
    plot_energy(particles, t, masses, plot_dir + name + "/")


def test_case_collision_angle():
    name = "test_case_collision_angle"
    xi, N, T, R = read_params(para_dir + name)
    a = 0.01
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
    plot_collision_angle(theta, bs, a, plot_dir + name + "/")


def profile_run():
    # https://web.archive.org/web/20140513005858im_/http://www.appneta.com/blog/line-profiler-python/
    name = "profile_run"
    xi, N, T, R = read_params(para_dir + name)
    radii = np.ones(N) * R
    masses = np.ones(N)
    args = (N, T, radii, masses, xi)
    particles, t = run_loop(random_dist, args)


def problem1(run_simulation = False):
    name = "problem1"
    path = data_dir + name + "/" 
    xi, N, T, R = read_params(para_dir + name)
    radii = np.ones(N) * R
    masses = np.ones(N)

    args = (N, T, radii, masses, xi)
    if run_simulation: simulate(path, random_dist, args)
    
    else:
        particles, t = read_data(path)
        dir = plot_dir + name + "/"
        plot_vel_dist(particles, 3*N, N, masses, dir)
        plot_av_vel(particles, dir)
        plot_particles(particles, -1, N, radii, dir)


def problem2(run_simulation=False):
    name = "problem2"
    path = data_dir + name + "/" 
    xi, N, T, R = read_params(para_dir + name)
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
    _, N, T, R = read_params(para_dir + name)
    xis = [1, 0.9, 0.8]
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
    xi, N, T, R = read_params(para_dir + name)

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


def problem4(i, j, run_simulation=False):
    name = "problem4"
    xi, N, T, R, all_Rs = read_params(para_dir + name)
    radii = np.ones(N) * R
    masses = np.ones(N) * R**2
    Rs = all_Rs[i:j]

    if run_simulation:
        for i, R in enumerate(Rs):
            radii[0] = R
            masses[0] = R**2
            args = (N, T, radii, masses, xi)

            path = data_dir + name + "/sweep_{}/".format(R)
            init = lambda N, radii : init_projectile(N, radii, 1)
            simulate(path, init, args, condition=energy_condition, n_check=100, TC=True)

    else:
        crater_size = np.zeros_like(Rs)
        for i, R in enumerate(Rs):
            radii[0] = R
            masses[0] = R**2
            args = (N, T, radii, masses, xi)

            path = data_dir + name + "/sweep_{}/".format(R)
            particles, t = read_data(path)
            dx = 0.015
            y_max = 0.5
            free_space = check_crater_size(particles, radii, -1, y_max, dx)
            crater_size[i] = dx**2 * np.sum(free_space)
            dir_path = "plots/" + name + "/"
            plot_particles(particles, -1, N, radii, dir_path, "particles{}".format(i))
            plot_crater(free_space, y_max, dir_path, "crater{}".format(i))
        
        plot_crater_size(Rs, crater_size, dir_path)


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
        i, j = int(args[2]), int(args[3])
        if args[-1] == "run":
            problem4(i, j, True)
        else:
            problem4(i, j)


if __name__ == "__main__":
    cl_arguments(sys.argv)
