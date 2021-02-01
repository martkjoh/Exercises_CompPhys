import numpy as np
from matplotlib import pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.animation import FuncAnimation as FA

from utillities import get_energy, get_temp, get_vel2, MaxBoltz


def plot_energy(particles, t, masses):
    fig, ax = plt.subplots()
    N = len(t)
    E = np.array([get_energy(particles, masses, n) for n in range(N)])
    T = len(particles)
    ax.plot(np.arange(T), E)
    plt.show()

def plot_vel_dist(particles, n0, dn, masses):
    fig, ax = plt.subplots()
    T = len(particles)
    N = len(particles[0])
    
    v2 = []
    n = n0
    temp = 0
    m = 0
    while n < T:
        v2.append(get_vel2(particles, n))
        temp += get_temp(particles, masses, n, N) 
        n += dn; m += 1

    temp = temp/m
    v2 = np.concatenate(v2)
    v = np.linspace(0, np.sqrt(np.max(v2)), 1000)
    print(len(v2))
    ax.plot(v, MaxBoltz(v, masses[0], temp))
    ax.hist(np.sqrt(v2), bins=30, density=True)

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
    