import numpy as np
from numpy.random import default_rng

rng = default_rng()
n = np.array([5, 6, 5, 3])
p0 = np.array([.45, .65, .35, .55])
p1 = np.array([0.1, 0.1, 0.1, 0.1])
p2 = 1 - p0 - p1

print(rng.multinomial(n, (p0, p1, p2)))
