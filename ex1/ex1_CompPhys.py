from matplotlib import collections
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.collections import PatchCollection
import heapq

from numpy.core.numeric import NaN


"""
Initializing
"""
# Sides of the box
L = 1
# Elasticity parametre
xi = 1
# Number of particles
N = 2
# Radii of the partiles
R = 0.1
# Number of timesteps
T = 10

radii = np.ones(N) * R
masses = np.ones(N)

# Particles must wholly inside the box, and not overlapping
def init_particles(N, radii):
    # particle_no, (x, y, vx, vy)
    particles = np.zeros((N, 4))
    i = 0
    k = 0
    while i<N:
        pos = np.random.rand(2)

        # Check if inside box
        if (pos[0] - radii[i]) < 0 or (pos[0] + radii[i]) > L: continue
        if (pos[1] - radii[i]) < 0 or (pos[1] + radii[i]) > L: continue

        # Check if overlap with other particles
        overlap = False
        for j in range(i):
            dist = (pos[0] - particles[j, 0])**2 + (pos[1] - particles[j, 1])**2
            if  dist < (radii[i] + radii[j])**2:
                overlap = True
                break
        
        if not overlap:
            vel = np.random.rand(2) - 0.5
            vel = vel/np.sqrt(vel[0]**2 + vel[1]**2)
            particles[i] = np.array([pos[0], pos[1], vel[0], vel[1]])
            i+=1

        # emergency break (heh)
        else: k += 1
        if k > 100: 
            raise Exception("can't fit particles")

    
    return particles

"""
Utillities
"""
    
def check_wall_collison(x, v, r):
    dt = np.inf
    if v > 0:
        dt = (L - r - x) / v
    elif v < 0:
        dt =(r - x) / v
    return dt

def find_wall_collisions(particles, collisions, t=0):
    for i in range(N):
        for j in range(2): # check both x and y wall
            dt = check_wall_collison(
                particles[i, 0+j], 
                particles[i, 2+j], 
                radii[i]
                )
            heapq.heappush(collisions, (t+dt, i, None, "wall"+str(j)))
    return collisions


def transelate(particles, n, dt):
    """ 
    Move the simulation forward for time dt.
    All particles are moved in a straight line.
    No collisions should occur during this time.
    """
    particles[n+1, :, :2] = particles[n, :, :2] + particles[n, :, 2:] * dt
    particles[n+1, :, 2:] = particles[n, :, 2:]



def push_next_collision(particles, i, t, collisions):
    collision_types = ["wall0", "wall1", "particle"]
    wall0 = check_wall_collison(particles[i, 0], particles[i, 2], radii[i])
    wall1 = check_wall_collison(particles[i, 1], particles[i, 3], radii[i])

    # Temp vals, untill particle-particle is implemented
    #TODO: impolement particle-particle collision
    particle = np.inf
    j = None

    collision_times = [wall0, wall1, particle]
    next_col = np.argmin(collision_times)
    heapq.heappush(collisions, (t+next_col, i, j, collision_types[next_col]))


def collide(particles, i, collision_type):
    if collision_type == "wall0":
        particles[i, 2:] = xi * np.array([-particles[i, 2], particles[i, 3]])
    if collision_type == "wall1":
        particles[i, 2:] = xi * np.array([particles[i, 2], -particles[i, 3]])
    elif collision_type == "particle":
        pass
    
    #TODO: implement particle-particle collisons

"""
main loop
"""

def run_loop():
    particles = np.empty((T+1, N, 4))
    particles[0] = init_particles(N, radii)
    collisions = []
    collisions = find_wall_collisions(particles[0], collisions)
    # When has particle i last collided? Used to remove invalid collisions
    last_collided = np.zeros(N)

    t = 0
    for n in range(T):
        next_coll = heapq.heappop(collisions)
        print(next_coll)
        plot_particles(particles[n])
        t_next = next_coll[0]
        i_next = next_coll[1]
        j_next = next_coll[2]
        # Skip invalid collisions
        if last_collided[i_next] > t_next: break

        dt = t_next - t
        transelate(particles, n, dt)
        collide(particles[n+1], i_next, next_coll[3])
        for i in [i_next, j_next]:
            if i==None: continue
            last_collided[i] = t_next
            push_next_collision(particles[n+1], i, t_next, collisions)

        t = t_next




"""
Plotting
"""

def plot_particles(particles, plot_vel=True):
    fig, ax = plt.subplots()
    circles = [
        plt.Circle((particles[i, 0], particles[i, 1]),
            radius=radii[i], 
            linewidth=0) 
        for i in range(N)
        ]        
    ax.add_collection(PatchCollection(circles))

    ax.set_ylim(0, 1)
    ax.set_xlim(0, 1)

    if plot_vel:
        length = 0.2
        [ax.arrow(
            particles[i, 0], 
            particles[i, 1], 
            particles[i, 2]*length, 
            particles[i, 3]*length,
            head_width=0.01)
            for i in range(N)]

    plt.show()

"""
Running
"""

run_loop()
