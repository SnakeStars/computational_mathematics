import numpy as np
import matplotlib.pyplot as plt

def phi(x):
    if x > l0 and x < l1:
        return 5
    return 0

def u(x, t, a, w, l, phi):
    return phi(np.mod(x - (a/w)*np.sin(w*t), l))

# Параметры

n = 200
a = 1
w = 1
l = 10
l0 = 2
l1 = 4

x = np.linspace(0, l, n)
h = x[1] - x[0]
t = h/(2*a)
tmax = 1000

# График

plt.ion()
fig, ax = plt.subplots()
line, = ax.plot(x, u(x, 1, a, w, l, phi))
ax.set_xlim(0, l)
ax.set_ylim(-10, 10)

for time in range(1, tmax):
    u_cur = u(x, time * t, a, w, l, phi)
    line.set_ydata(u_cur)
    plt.pause(0.01)
plt.ioff()