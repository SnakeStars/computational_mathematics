import numpy as np
import matplotlib.pyplot as plt

# Параметры

n = 200
a = 1
w = 1
l = 10

x = np.linspace(0, l, n)
h = x[1] - x[0]
t = h/(2*a)
tmax = 1000

U = np.zeros((2,n))
U[1] = 5*np.sin(2*np.pi*x/l)

# График

plt.ion()
fig, ax = plt.subplots()
line, = ax.plot(x, U[1])
ax.set_xlim(0, l)
ax.set_ylim(-10, 10)

for time in range(1, tmax):
    A = a*np.cos(w*time*t)
    if A > 0:
        U[0][:] = U[1][:] - (A*t/h) * (U[1][:] - np.roll(U[1], 1))
    else:
        U[0][:] = U[1][:] - (A*t/h) * (np.roll(U[1], -1) - U[1][:])
    line.set_ydata(U[1])
    U[1] = U[0].copy()
    plt.pause(0.01)