import numpy as np
from matplotlib import pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.animation import FuncAnimation as FA

from utillities import get_energy, get_temp, get_vel2, MaxBoltz, check_dir


def save_plot(fig, ax, fname, dir_path):
    check_dir(dir_path)
    plt.savefig(dir_path + fname + ".pdf")
    plt.cla()


def plot_energy(particles, t, masses, dir_path):
    fig, ax = plt.subplots()
    N = len(t)
    E = np.array([get_energy(particles, masses, n) for n in range(N)])
    T = len(particles)
    ax.plot(np.arange(T), E)
    save_plot(fig, ax, "energy", dir_path)


def plot_vel_dist(particles, n0, dn, masses, dir_path):
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
 
    ax.plot(v, MaxBoltz(v, masses[0], temp))
    ax.hist(np.sqrt(v2), bins=20, density=True)
    ax.set_title("$T={}$".format(temp))

    save_plot(fig, ax, "vel_dist", dir_path)



def plot_collision_angle(theta, bs, a, dir_path):
    fig, ax = plt.subplots()
    ax.plot(theta, bs)
    ax.plot(theta, a  *  np.sin(theta / 2), "k--")
    
    save_plot(fig, ax, "collision_angle", dir_path)


def plot_energy_prob3(particles, t, masses, N1, N2, dir_path="plots/"):
    fig, ax = plt.subplots()
    T = len(t)
    E1 = np.array([get_energy(particles[:, :N1], masses[:N1], n) for n in range(T)]) / N1
    E2 = np.array([get_energy(particles[:, N1:], masses[N1:], n) for n in range(T)])  / N2
    Etot = np.array([get_energy(particles, masses, n) for n in range(T)]) / (N1 + N2)

    ax.plot(np.arange(T), E1, label="$m = 1$")
    ax.plot(np.arange(T), E2, label="$m = 4$")
    ax.plot(np.arange(T), Etot, label="All")
    ax.legend()

    save_plot(fig, ax, "energy_ex3", dir_path)


def plot_crater(free_space, y_max, dir_path, fname):
    Nx, Ny = np.shape(free_space)
    x = np.linspace(0, 1, Nx)
    y = np.linspace(0, y_max, Ny)
    x, y = np.meshgrid(x, y)
    
    fig, ax = plt.subplots()
    ax.imshow(free_space.T[::-1])
    save_plot(fig, ax, fname, dir_path)
    

def plot_crater_size(Rs, crater_sizes, dir_path):
    fig, ax = plt.subplots()
    ax.plot(Rs, crater_sizes, "x")
    save_plot(fig, ax, "crater_size", dir_path)


def get_particles_plot(particles, n, N, radii):
    circles =  [plt.Circle(
        (particles[n, i, 0], particles[n, i, 1]), radius=radii[i], linewidth=0) 
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


def plot_particles(particles, n, N, radii, dir_path, fname="particles"):
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

    save_plot(fig, ax, fname, dir_path)


def anim_particles(particles, t, N, radii, dt, intr=100, title="vid", plot_vel=True):
    fig, ax = plt.subplots()
    ax.set_ylim(0, 1)
    ax.set_xlim(0, 1)

    circles = get_particles_plot(particles, 0, N, radii)
    arrows = get_arrows_plot(particles, 0, N, radii)

    patches = PatchCollection(circles + arrows)
    colors = np.concatenate([np.linspace(0.2, 0.8, N), np.zeros(N)])
    patches.set_array(colors)
    ax.add_collection(patches)
    txt1 = ax.text(0.8, 0.8, "t = {:.3f}".format(t[0]))
    txt2 = ax.text(0.8, 0.9, "n = {}/{}".format(0, len(t)))
    ax.text(0.8, 0.75, "t_f = {:.3f}".format(t[-1]))

    steps = np.nonzero(np.diff(t//dt))[0]
    frames = len(steps)

    print("writing {} frames".format(frames))
    def anim(n):
        n = steps[n]
        txt1.set_text("t = {:.3f}".format(t[n]))
        txt2.set_text("n = {}/{}".format(n, len(t)))
        circles = get_particles_plot(particles, n, N, radii)
        arrows = get_arrows_plot(particles, n, N, radii)
        patches.set_paths(circles + arrows)

    a = FA(fig, anim, interval=intr, frames=frames)
    a.save("video/" + title + ".mp4", dpi=300)
