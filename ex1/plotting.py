import numpy as np
from matplotlib import pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.animation import FuncAnimation as FA

from utillities import get_energy, get_temp, get_vel2, MaxBoltz, check_dir

plt.rcParams['mathtext.fontset'] = 'cm'
font = {'family' : 'serif', 
        'size': 20}
plt.rc('font', **font)
plt.rc('lines', lw=2)


def save_plot(fig, ax, fname, dir_path):
    check_dir(dir_path)
    plt.tight_layout()
    plt.savefig(dir_path + fname + ".pdf")
    plt.close(fig)


def plot_energy(particles, t, masses, dir_path):
    fig, ax = plt.subplots(figsize=(10, 7))
    N = len(t)
    E = np.array([get_energy(particles, masses, n) for n in range(N)])
    ax.set_xlabel("$t$")
    ax.set_ylabel("$E$")
    ax.plot(t, E, label="$E_\mathrm{tot}$")
    ax.legend()
    save_plot(fig, ax, "energy", dir_path)


def plot_av_vel(particles, args, dir_path):
    fig, ax = plt.subplots(figsize=(12, 5))
    N, T, radii, masses, xi, N_save = args

    N = len(particles[0])
    v = particles[:, :, 2:]
    v_av = np.einsum("tn -> t", np.sqrt(v[:, :, 0]**2 + v[:, :, 1]**2)) / N
    t = np.linspace(0, T, N_save)
    ax.plot(t, v_av)
    ax.set_xlabel("# collisions")
    ax.set_ylabel("$\\langle v \\rangle$")
    save_plot(fig, ax, "v_av", dir_path)


def get_plot_vel_dist(ax, particles, masses, title, n0, dn, bins=100, graph=True):
    T = len(particles)
    N = len(particles[0])
    
    v2 = []; temp = 0; m = 0; n = n0
    while n < T:
        v2.append(get_vel2(particles, n))
        temp += get_temp(particles, masses, n, N) 
        n += dn; m += 1

    temp = temp/m
    v2 = np.concatenate(v2)
    v = np.linspace(0, np.sqrt(np.max(v2)), 1000)

    ax.hist(np.sqrt(v2), bins=bins, density=True)
    ax.set_title(title + "$T = {:.3f} ,\, N = {}$".format(temp, N))
    ax.set_xlabel("$v$")
    ax.set_ylabel("prob.dens.")
    
    if graph:
        label = "$f(v) = \\frac{mv}{T} \exp\left( - \\frac{m v^2}{2 T}\\right)$"
        ax.plot(v, MaxBoltz(v, masses[0], temp), label=label)
        ax.legend()
    

def plot_vel_dist(
        particles, masses, dir_path, n0, dn, bins=100, graph=True, title="", fname="vel_dist"
        ):
    fig, ax = plt.subplots(figsize=(8, 5))
    get_plot_vel_dist(ax, particles, masses, title, n0, dn, bins=bins, graph=graph)

    save_plot(fig, ax, fname, dir_path)


def plot_prob_2(particles, args, N1, t, dir_path, titles, fname):
    N, T, radii, masses, xi, N_save = args
    skip = (T-1)//(N_save-1)

    fig, ax = plt.subplots(2, 2, figsize=(18, 8), sharex=True)
    v = np.sqrt(get_vel2(particles, -1))
    bins = np.linspace(np.min(v), np.max(v), 100)
    get_plot_vel_dist(ax[0, 0], particles[0:1, :N1], masses[:N1], titles[0], 0, 1,graph=False, bins=bins)
    get_plot_vel_dist(ax[0, 1], particles[0:1, N1:], masses[N1:], titles[1], 0, 1, graph=False, bins=bins)

    get_plot_vel_dist(ax[1, 0], particles[:, :N1], masses[:N1], titles[0], n0=3*N//skip, dn=N//skip)
    get_plot_vel_dist(ax[1, 1], particles[:, N1:], masses[N1:], titles[1], n0=3*N//skip, dn=N//skip)
    
    save_plot(fig, ax, fname, dir_path)

    fig, ax = plt.subplots(figsize=(12, 5))

    N = len(particles[0])
    v = particles[:, :, 2:]
    v_av1 = np.einsum("tn -> t", np.sqrt(v[:, :N1, 0]**2 + v[:, :N1, 1]**2)) / N1
    v_av2 = np.einsum("tn -> t", np.sqrt(v[:, N1:, 0]**2 + v[:, N1:, 1]**2)) / (N - N1)
    ax.plot(t, v_av1, label="$m=1$")
    ax.plot(t, v_av2, label="$m=4$")
    ax.legend()
    ax.set_xlabel("$t$")
    ax.set_ylabel("$\\langle v \\rangle$")
    
    save_plot(fig, ax, "v_av", dir_path)


def plot_collision_angle(theta, bs, a, dir_path):
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(theta, bs, "bx", label="$\\theta_m$")
    ax.plot(theta, a * np.cos(theta / 2), "k-", label="$a \cos(\\theta/2)$")
    ax.set_xlabel("$\\theta$")
    ax.set_ylabel("$s$")
    ax.legend()
    
    save_plot(fig, ax, "collision_angle", dir_path)


def plot_energy_prob3(particles, t, masses, N1, N2, dir_path="plots/"):
    fig, ax = plt.subplots(figsize=(7, 5))
    T = len(t)
    E1 = np.array([get_energy(particles[:, :N1], masses[:N1], n) for n in range(T)]) / N1
    E2 = np.array([get_energy(particles[:, N1:], masses[N1:], n) for n in range(T)])  / N2
    Etot = np.array([get_energy(particles, masses, n) for n in range(T)]) / (N1 + N2)

    ax.plot(t, E1, label="$m = 1,$")
    ax.plot(t, E2, label="$m = 4,$")
    ax.plot(t, Etot, label="Total")
    ax.set_ylabel("$\\langle E \\rangle$")
    ax.set_xlabel("$t$")
    ax.legend()

    save_plot(fig, ax, "energy_ex3", dir_path)


def plot_crater(free_space, dir_path, fname):
    Nx, Ny = np.shape(free_space)
    x = np.linspace(0, 1, Nx)
    y = np.linspace(0, 0.5, Ny)
    x, y = np.meshgrid(x, y)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(free_space.T[::-1])
    save_plot(fig, ax, fname, dir_path)
    

def plot_crater_size(ms, crater_sizes, dir_path):
    av_rise = np.sum(crater_sizes/ ms) / len(ms)
    r = np.linspace(0, np.max(ms))
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(ms, crater_sizes, "rx", label="Sampled sizes")
    ax.plot(r, av_rise * r, "k--", label="${:.3f} m$".format(av_rise))
    ax.legend()
    ax.set_xlabel("$m$")
    ax.set_ylabel("Crater area")
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
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_ylim(0, 1)
    ax.set_xlim(0, 1)
    circles  = get_particles_plot(particles, n, N, radii)
    arrows = get_arrows_plot(particles, n, N, radii)
    patches = PatchCollection(circles + arrows)
    v = np.sqrt(particles[n, :, 2]**2 + particles[n, :, 3]**2)
    vmax = np.max(v)
    if vmax<1e-10: vmax = 1 # avoid divide by zero
    colors = np.concatenate([v/vmax, np.zeros(N)])
    patches.set_array(colors)
    ax.add_collection(patches)

    save_plot(fig, ax, fname, dir_path)


def anim_particles(particles, t, N, radii, dt, interval=100, title="vid", plot_vel=True):
    fig, ax = plt.subplots()
    ax.set_ylim(0, 1)
    ax.set_xlim(0, 1)

    circles = get_particles_plot(particles, 0, N, radii)
    arrows = get_arrows_plot(particles, 0, N, radii)

    patches = PatchCollection(circles + arrows)
    v = np.sqrt(particles[0, :, 2]**2 + particles[0, :, 3]**2)
    vmax = np.max(v)
    if vmax<1e-10: vmax = 1 # avoid divide by zero
    colors = np.concatenate([v/vmax, np.zeros(N)])
    patches.set_array(colors)
    ax.add_collection(patches)
    txt1 = ax.text(0.8, 0.8, "t = {:.3f}".format(t[0]), size=12)
    txt2 = ax.text(0.8, 0.9, "n = {}/{}".format(0, len(t)), size=12)
    ax.text(0.8, 0.75, "t_f = {:.3f}".format(t[-1]), size=12)

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
        v = np.sqrt(particles[n, :, 2]**2 + particles[n, :, 3]**2)
        vmax = np.max(v)
        if vmax<1e-10: vmax = 1 # avoid divide by zero
        colors = np.concatenate([v/vmax, np.zeros(N)])
        patches.set_array(colors)

    a = FA(fig, anim, interval=interval, frames=frames)
    a.save("video/" + title + ".mp4", dpi=300)
