import numpy as np
import matplotlib.pyplot as plt

def phi(x):
    return x

def u(x, t, a, w, l, phi):
    return phi(np.mod(x - (a/w)*np.sin(w*t), l))

# Параметры

n = 200
a = 1
w = 1
l = 10
