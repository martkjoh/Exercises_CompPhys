import numpy as np
from scipy.sparse.linalg import splu
from scipy.sparse import diags, csc_matrix
from matplotlib import pyplot as plt
from scipy.integrate import simps

# Diffusion parameter
K = 10

# Specify domain
Xmin = 0
Xmax = 100
Nx = 10_001
X, dx = np.linspace(Xmin, Xmax, Nx, retstep = True)

Tmax = 100
dt = 0.1
Nt = int(Tmax/dt)


alpha = K*dt/(2*dx**2)
a = alpha

V0 = -4*a*np.ones_like(Nx)
V1 = 2 * a * np.ones(Nx-1)
V2 = 2 * a * np.ones(Nx-1)
V1[0] = 4*a
V2[-1] = 4*a
D = csc_matrix(diags((V2, V0, V1), (-1, 0, 1)))

I = csc_matrix(diags(np.ones(Nx)))
L = I - D/2
R = I + D/2


z = np.linspace(Xmin, Xmax, Nx)
C0 = np.exp(-(z - Xmax/2)**2/(2 * 1/20)**2)

C = np.zeros((Nt+1, Nx))
C[0,:] = C0

LU = splu(L)
for i in range(1, Nt+1):
    x = R.dot(C[i-1,:])
    C[i,:] = LU.solve(x)


fig = plt.figure(figsize = (9, 5))
mass = simps(C, x = X, axis = 1)
times = np.linspace(0, Tmax, len(mass))
plt.plot(times, mass - mass[0])
plt.show()