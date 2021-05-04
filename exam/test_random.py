import numpy as np
from numpy.random import default_rng

rng = default_rng()
B = rng.binomial
M = rng.multinomial

n = np.array([
    [100, 5],
    [100, 0],
    [1000, 200]
])

p = np.array([.9, .2])

# print(B(n, p))
print(B(n, p[0]))
print(np.moveaxis(M(n, (p[0], 1-p[0])), -1, 0)[0, :, :])


# x = n[:, np.newaxis]
# print((x).dtype)
# print(x)