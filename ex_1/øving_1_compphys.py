from matplotlib import collections
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.collections import PatchCollection
import heapq


"""
Initializing
"""
# Sides of the box
L = 1
# Elasticity parametre
xi = 1
# Number of particles
N = 10
# Radii of the partiles
R = 0.1
# Number of timesteps
T = 1

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
            vel = np.random.rand(2)
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
        (r - x) / v
    return dt

def find_wall_collisions(particles, t=0):
    collisions = []
    for i in range(N):
        for j in range(2): # check both x and y wall
            dt = check_wall_collison(
                particles[t, i, 0+j], 
                particles[t, i, 2+j], 
                radii[i]
                )
            heapq.heappush(collisions, (dt, i, "wall"+str(j)))
    return collisions


"""
Plotting
"""

def plot_particles(particles):
    fig, ax = plt.subplots()
    circles = [
        plt.Circle((particles[i, 0], particles[i, 1]),
            radius=radii[i], 
            linewidth=0) 
        for i in range(N)
        ]        
    ax.add_collection(PatchCollection(circles))

    plt.show()

"""
Running
"""

particles = init_particles(N, radii)
plot_particles(particles)
