#!/usr/bin/env python3

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
#TODO: run profiler

def test_case_one_particle():
    name = "test_case_one_particle"
    xi, N, T, R, N_save = read_params(para_dir + name)
    print(N_save)
    radii = np.ones(N) * R
    masses = np.ones(N)
    args = (N, T, radii, masses, xi, N_save)

    particles, t = run_loop(init_one_testparticle, args)
    
    for i in [-4, -3, -2, -1]:
        plot_particles(particles, i, N, radii, plot_dir + name + "/", "particle{}".format(i))
    plot_energy(particles, t, masses, plot_dir + name + "/")


def test_case_two_particles():
    name = "test_case_two_particles"
    xi, N, T, R, N_save = read_params(para_dir + name)
    radii = np.ones(N) * R
    masses = np.ones(N)
    args = (N, T, radii, masses, xi, N_save)

    particles, t = run_loop(init_two_testparticles, args)
    anim_particles(particles, t, N, radii, 0.001, title=name)
    plot_energy(particles, t, masses, plot_dir + name + "/")


def test_case_many_particles():
    name = "test_case_many_particles"
    xi, N, T, R, N_save = read_params(para_dir + name)
    radii = np.ones(N) * R
    masses = np.ones(N)
    args = (N, T, radii, masses, xi, N_save)

    particles, t = run_loop(random_dist, args)
    anim_particles(particles, t, N, radii, 0.0005, interval=100, title=name)
    plot_particles(particles, -1, N, radii, plot_dir + name + "/", name)
    plot_energy(particles, t, masses, plot_dir + name + "/")


def test_case_collision_angle():
    name = "test_case_collision_angle"
    xi, N, T, R, N_save = read_params(para_dir + name)
    a = 0.01
    radii = np.array([a, R])
    masses = np.array([1e6, 1])
    args = (N, T, radii, masses, xi, N_save)

    m = 100
    bs = np.linspace(0, a, m)
    theta = np.empty(m)
    for i, b in enumerate(bs):
        init = lambda N, radii : init_collision_angle(b, N, radii)
        particles, t = run_loop(init, args)
        x, y = particles[2, 1, :2]
        x -= 0.5
        y -= 0.5
        theta[i] = np.arctan2(y, x)
    
    plot_collision_angle(theta, bs, a, plot_dir + name + "/")


def test_case_projectile(run_simulation=False):
    name = "test_case_projectile"
    xi, N, T, R, N_save = read_params(para_dir + name)
    radii = np.ones(N) * R
    radii[0] = 0.05
    masses = np.ones(N)
    masses[0] = 25
    args = (N, T, radii, masses, xi, N_save)

    init = lambda N, radii : init_projectile(N, radii, 5)
    particles, t = run_loop(init, args, TC=True)
    free_space = check_crater_size(particles[:, 1:], radii, -1, 180)
    dir_path = "plots/" + name + "/"
    plot_particles(particles, -1, N, radii, dir_path, name + "_particles")
    plot_crater(free_space, dir_path, name +"_crater")
    anim_particles(particles, t, N, radii, 0.005, title=name)
    


def profile_run():
    # https://web.archive.org/web/20140513005858im_/http://www.appneta.com/blog/line-profiler-python/
    name = "profile_run"
    xi, N, T, R, N_save = read_params(para_dir + name)
    radii = np.ones(N) * R
    masses = np.ones(N)
    args = (N, T, radii, masses, xi, N_save)
    particles, t = run_loop(random_dist, args)


def problem1(run_simulation = False):
    name = "problem1"
    path = data_dir + name + "/" 
    xi, N, T, R, N_save = read_params(para_dir + name)
    skip = (T-1)//(N_save-1) # collisions bw every saved event
    start = 3*N//skip # start sampling
    radii = np.ones(N) * R
    masses = np.ones(N)
    args = (N, T, radii, masses, xi, N_save)

    if run_simulation:
        particles, t = run_loop(random_dist, args)
        save_data(particles, t, path, 1)

    particles, t = read_data(path)
    dir = plot_dir + name + "/"

    v = np.sqrt(get_vel2(particles, -1))
    bins = np.linspace(np.min(v), np.max(v), 100)
    plot_vel_dist(particles[0:1], masses, dir + "2/", 0, 1, graph=False, bins=bins)
    plot_vel_dist(particles, masses, dir, 3*N//skip, N//skip, bins=100)
    plot_av_vel(particles, args, dir)
    plot_particles(particles, -1, N, radii, dir)


def problem2(run_simulation=False):
    name = "problem2"
    path = data_dir + name + "/" 
    xi, N, T, R, N_save = read_params(para_dir + name)
    skip = (T-1)//(N_save-1)
    radii = np.ones(N) * R
    masses = np.empty(N)
    N1 = N//2
    N2 = N - N1
    masses[:N//2] = np.ones(N1)
    masses[N//2:] = 4 * np.ones(N2)
    args = (N, T, radii, masses, xi, N_save)

    if run_simulation:
        particles, t = run_loop(random_dist, args)
        save_data(particles, t, path, skip)

    particles, t = read_data(path)
    dir = plot_dir + name + "/"
    titles = ("$m = 1,$", "$m = 4,$")
    plot_prob_2(particles, args, N1, t, dir, titles, "vel_dist")


def problem3(run_simulation=False):
    name = "problem3"
    path = data_dir + name + "/" 
    _, N, T, R, N_save = read_params(para_dir + name)
    skip = (T-1)//(N_save-1)
    xis = [1, 0.9, 0.8]
    radii =np.ones(N) * R
    masses = np.empty(N)
    N1 = N//2
    N2 = N - N1
    masses[:N//2] = np.ones(N1)
    masses[N//2:] = 4 * np.ones(N2)

    for i, xi in enumerate(xis):
        if run_simulation: 
            path_xi = path + "xi_" + str(i) + "/"
            args = (N, T, radii, masses, xi, N_save)
            particles, t = run_loop(random_dist, args, TC=True)
            save_data(particles, t, path_xi, skip)

        dir = "xi_" + str(i) + "/"
        path_xi = path + dir
        particles, t = read_data(path_xi)
        plot_energy_prob3(particles, t, masses, N1, N2, plot_dir + name + "/" + dir)

    
def problem4(i, j, run_simulation=False):
    name = "problem4"
    xi, N, T, R, N_save, N_R = read_params(para_dir + name)
    skip = (T-1)//(N_save-1)
    radii = np.ones(N) * R
    masses = np.ones(N) * R**2
    Rs = np.linspace(0.01, 0.04, int(N_R))
    crater_size = np.zeros_like(Rs)
    all = i==0 and j==N_R

    for i in range(i, j):
        R = Rs[i]
        radii[0] = R
        masses[0] = R**2
        args = (N, T, radii, masses, xi, N_save)
        path = data_dir + name + "/sweep_{}/".format(i)

        if run_simulation:
            init = lambda N, radii : init_projectile(N, radii, 1)
            particles, t = run_loop(init, args, TC=True, condition=energy_condition)
            save_data(particles, t, path, skip)

        if all:
            particles, t = read_data(path)
            dx = 0.012
            y_max = 0.5
            free_space = check_crater_size(particles, radii, -1, y_max, dx)
            crater_size[i] = 0.5 - dx**2 * np.sum(free_space)
            dir_path = "plots/" + name + "/"
            plot_particles(particles, -1, N, radii, dir_path, "particles{}".format(i))
            plot_crater(free_space, y_max, dir_path, "crater{}".format(i))
        
    if all: plot_crater_size(Rs, crater_size, dir_path)



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
    if args[1] == "test":
        for arg in args[2:]:
            tests[int(arg)]()

    elif args[1] == "problem":
        for arg in args[2:]:
            try: int(arg)
            except: break
            problems[int(arg)](args[-1] == "run")

    elif args[1] == "sweep":
        i, j = int(args[2]), int(args[3])
        problem4(i, j, args[-1]=="run")


if __name__ == "__main__":
    cl_arguments(sys.argv)

    """
    To run a the program, use commands "python ./main.py " then
    - "test 0 1 2" to run simulation and plot stored data for test case 0, 1, and 2
    - "problem 0 1 (run)" to (run simulation and) plot stored data for problem1, problem2
    - "sweep 0 10 (run)" to (run simulations and) plot sored data with R-valuse nr 0 to (not incl) 10
    to run sweep in parallel, execute for example
    "sweep run 0 4", "sweep run 4 8", etc. in different terminal. The first R's take a lot less time.
    """
