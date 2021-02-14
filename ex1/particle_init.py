import numpy as np


# Sides of the box
L = 1 

# Functions for preparing the inital particle distributions
def init_one_testparticle(N, radii):
    R = radii[0]
    return np.array([[R, 1/2, 1, -1],])


def init_two_testparticles(N, radii):
    R = radii[0]
    return np.array([
        [R, 0.5, 1, 0],
        [0.5, 1-R, 0, -1]])


def init_collision_angle(b, N, radii):
    return np.array([
        [1/2, 1/2, 0, 0],
        [radii[1], 1/2 + b, 1, 0]
    ])


# Particles must wholly inside the box, and not overlapping
def random_dist(N, radii, x=(0, 1), y=(0, 1), v=1):
    # particle_no, (x, y, vx, vy)
    particles = np.zeros((N, 4))
    i = 0
    k = 0
    while i<N:
        pos = np.random.rand(2)

        # Check if inside box
        if (pos[0] - radii[i]) < x[0] or (pos[0] + radii[i]) > x[1]: continue
        if (pos[1] - radii[i]) < y[0] or (pos[1] + radii[i]) > y[1]: continue

        # Check if overlap with other particles
        overlap = False
        for j in range(i):
            dist = (pos[0] - particles[j, 0])**2 + (pos[1] - particles[j, 1])**2
            if  dist < (radii[i] + radii[j])**2:
                overlap = True
                break
        
        if not overlap:
            vel = np.random.rand(2) - 0.5
            vel = vel/np.sqrt(vel[0]**2 + vel[1]**2) * v
            particles[i] = np.array([pos[0], pos[1], vel[0], vel[1]])
            i+=1
            k = 0

        # emergency break (heh)
        else: k += 1
        if k > 200: 
            raise Exception("can't fit particles")
    
    return particles


def init_projectile(N, radii, v_proj):
    particles = random_dist(N-1, radii[1:], x=(0, 1), y=(0, 0.5), v=0)
    return np.concatenate([np.array([[0.5, 0.75, 0, -1 * v_proj],]), particles])
