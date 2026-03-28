import numpy as np
import matplotlib.pyplot as plt

# Параметры

n = 200
a = 1
w = 1
l = 10

x = np.linspace(0, l, n+1)[:-1]
h = x[1] - x[0]
x_center = x + h/2
t = h/(2*a)
tmax = 300

U = np.zeros((2,n))
U[1] = 5*np.sin(2*np.pi*x/l)

F = np.zeros((2,n))
F[1][:] = a

U_wave = np.zeros((n))

# График

plt.ion()
fig, ax = plt.subplots()
line, = ax.plot(x, U[1])
ax.set_xlim(0, l)
ax.set_ylim(-10, 10)

for time in range(1, tmax):

    U[0][:] = U[1][:] - (t/h)*(F[1][:] - np.roll(F[1], 1))

    # print(F[1][:])
    # print(np.roll(F[1], 1))

    A = a*np.cos(w*(time + 1/2)*t)
    if A >= 0:
        #F[0][:] = F[1][:] - (A*t/h) * (U[1][:] - np.roll(U[1], 1))
        U_wave = U[1][:]
        F[0][:] = A * U_wave

    else:
        #F[0][:] = F[1][:] - (A*t/h) * (np.roll(U[1], -1) - U[1][:])
        U_wave = np.roll(U[1], -1)
        F[0][:] = A * U_wave
    line.set_ydata(U[1])
    U[1] = U[0].copy()
    F[1] = F[0].copy()
    plt.pause(0.01)