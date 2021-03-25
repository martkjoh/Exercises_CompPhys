import numpy as np
from utillities import *
from plot import *

def prob2():
    def K(z, args):
        K0 = 1e-3
        Ka = 2e-2
        za = 7
        Kb = 5e-2
        zb = 10
        return K0 + Ka * z / za  *np.exp(z / za) + Kb * (L - z)/zb * np.exp(-(L - z)/zb)