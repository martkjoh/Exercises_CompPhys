from matplotlib.pyplot import title
import numpy as np
import sys

from utillities import check_crater_size, read_data, run_loop, save_data, energy_condition, read_params
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
    
    for i in [-4, -3, -2, -1]:
        plot_particles(particles, i, N, radii, plot_dir + name + "/", "particle{}".format(i))
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
    # anim_particles(particles, t, N, radii, 0.001, intr=100, title=name)
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


def test_case_projectile(run_simulation=False):
    name = "test_case_projectile"
    xi, N, T, R = read_params(para_dir + name)
    radii = np.ones(N) * R
    radii[0] = 0.05
    masses = np.ones(N)
    masses[0] = 25
    args = (N, T, radii, masses, xi)

    path = data_dir + name + "/"
    N_save = 1000
    skip = T//N_save

    if run_simulation:
        init = lambda N, radii : init_projectile(N, radii, 5)
        particles, t = run_loop(init, args, TC=True)
        save_data(particles, t,  path, skip)

    else:
        particles, t = read_data(path)
        anim_particles(particles, t, N, radii, 0.005, title=name)


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

    N_save = 200 # number of steps to save (+- 1)
    skip = T//N_save # values to skip to save N_save values

    if run_simulation: 
        particles, t = run_loop(random_dist, args)
        save_data(particles, t, path, skip)

    else:
        particles, t = read_data(path)
        dir = plot_dir + name + "/"
        start = 3*N // skip
        plot_vel_dist(particles[start:], masses, dir)
        v = np.sqrt(get_vel2(particles, -1))
        bins = np.linspace(np.min(v), np.max(v), 100)
        plot_vel_dist(particles[0:1], masses, dir + "2/", graph=False, bins=bins)
        plot_av_vel(particles, T, skip, dir)
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

    N_save = 200
    skip = T//N_save

    if run_simulation:
        particles, t = run_loop(random_dist, args)
        save_data(particles, t, path, skip)

    else:
        particles, t = read_data(path)
        start = 3*N // skip
        dir = plot_dir + name + "/"
        titles = ("$m = 1,$", "$m = 4,$")
        plot_prob_2(particles, start, N1, masses, t, dir, titles, "vel_dist")


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

    N_save = 200
    skip = T//N_save

    if run_simulation: 
        for i, xi in enumerate(xis):
            path_xi = path + "xi_" + str(i) + "/"
            args = (N, T, radii, masses, xi)
            particles, t = run_loop(random_dist, args, TC=True)
            save_data(particles, t, path_xi, skip)

    else:
        for i, xi in enumerate(xis):
            dir = "xi_" + str(i) + "/"
            path_xi = path + dir
            particles, t = read_data(path_xi)
            plot_energy_prob3(particles, t, masses, N1, N2, plot_dir + name + "/" + dir)

    
def problem4(i, j, run_simulation=False):
    name = "problem4"
    xi, N, T, R, all_Rs = read_params(para_dir + name)
    radii = np.ones(N) * R
    masses = np.ones(N) * R**2
    Rs = all_Rs[i:j]

    N_save = 1
    skip = T//N_save

    if run_simulation:
        for i, R in enumerate(Rs):
            radii[0] = R
            masses[0] = R**2
            args = (N, T, radii, masses, xi)

            path = data_dir + name + "/sweep_{}/".format(R)
            init = lambda N, radii : init_projectile(N, radii, 1)
            particles, t = run_loop(init, args, TC=True, n_check=100, condition=energy_condition)
            save_data(particles[-1][None], t, path, skip)
            # The None is a hack to make sure the list have the right amount of indices

    else:
        crater_size = np.zeros_like(Rs)
        for i, R in enumerate(Rs):
            radii[0] = R
            masses[0] = R**2
            args = (N, T, radii, masses, xi)

            path = data_dir + name + "/sweep_{}/".format(R)
            particles, t = read_data(path)
            dx = 0.012
            y_max = 0.5
            free_space = check_crater_size(particles, radii, -1, y_max, dx)
            crater_size[i] = 0.5 - dx**2 * np.sum(free_space)
            dir_path = "plots/" + name + "/"
            plot_particles(particles, 0, N, radii, dir_path, "particles{}".format(R))
            plot_crater(free_space, y_max, dir_path, "crater{}".format(R))
        
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
                if arg == "run" : break # not very elegant
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
