import numpy as np
import matplotlib.pyplot as plt

def phi2(x, l0, l1):
    mask = (x >= l0) & (x <= l1)
    return mask.astype(float) * 5

def phi(x):
    return 5*np.sin(2*np.pi*x/l)

def u(x, t, a, w, l, phi):
    return phi(np.mod(x - (a/w)*np.sin(w*t), l))

def u2(x, t, a, w, l, phi, l0, l1):
    return phi(np.mod(x - (a/w)*np.sin(w*t), l), l0, l1)

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
tmax = 500

U = np.zeros((2,n))
U[1] = 5*np.sin(2*np.pi*x/l)

# График

plt.ion()
fig, ax = plt.subplots()
line, = ax.plot(x, U[1])
line2, = ax.plot(x, u(x, 1, a, w, l, phi), color="red", linestyle='--')
ax.set_xlim(0, l)
ax.set_ylim(-10, 10)

for time in range(1, tmax):
    A = a*np.cos(w*time*t)
    if A > 0:
        U[0][:] = U[1][:] - (A*t/h) * (U[1][:] - np.roll(U[1], 1))
    else:
        U[0][:] = U[1][:] - (A*t/h) * (np.roll(U[1], -1) - U[1][:])
    
    line.set_ydata(U[1])
    line2.set_ydata(u(x, time*t, a, w, l, phi))

    U[1] = U[0].copy()
    plt.pause(0.01)


plt.pause(1)
ax.clear()

U = np.zeros((2,n))
U[1] = phi2(x, l0, l1)

line, = ax.plot(x, U[1])
line2, = ax.plot(x, u2(x, 1, a, w, l, phi2, l0, l1), color="red", linestyle='--')
ax.set_xlim(0, l)
ax.set_ylim(-10, 10)

for time in range(1, tmax):
    A = a*np.cos(w*time*t)
    if A > 0:
        U[0][:] = U[1][:] - (A*t/h) * (U[1][:] - np.roll(U[1], 1))
    else:
        U[0][:] = U[1][:] - (A*t/h) * (np.roll(U[1], -1) - U[1][:])
    
    line.set_ydata(U[1])
    line2.set_ydata(u2(x, time*t, a, w, l, phi2, l0, l1))

    U[1] = U[0].copy()
    plt.pause(0.25)