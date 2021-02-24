import numpy as np

def heun_step(f, y, n, h):
    y[n+1] = y[n] + h * f(y[n])
    y[n+1] = y[n] + (h / 2) * (f(y[n]) + f(y[n+1]))

def LLG()