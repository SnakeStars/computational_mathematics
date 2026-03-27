import numpy as np
import matplotlib.pyplot as plt

def phi(x):
    return 5*np.sin(2*np.pi*x/l)

def u(x, t, a, w, l, phi):
    return phi(np.mod(x - (a/w)*np.sin(w*t), l))

# Параметры

n = 200
a = 1
w = 1
l = 10

x = np.linspace(0, l, n)
h = x[1] - x[0]
# t = h/(2*a)
ts = h/(2*a)
T = 1
# tmax = 1000

errors = []

# График

# plt.ion()
# fig, ax = plt.subplots()
# line, = ax.plot(x, U[1])
# analytical_line, = ax.plot(x, u(x, 1, a, w, l, phi), color='red', linestyle='--')
# ax.set_xlim(0, l)
# ax.set_ylim(-10, 10)

for t in ts:

    U = np.zeros((2,n))
    U[1] = 5*np.sin(2*np.pi*x/l)

    error = 0
    max_error = 0

    tmax = int(T/t)

    for time in range(1, tmax):
        A = a*np.cos(w*time*t)
        if A > 0:
            U[0][:] = U[1][:] - (A*t/h) * (U[1][:] - np.roll(U[1], 1))
        else:
            U[0][:] = U[1][:] - (A*t/h) * (np.roll(U[1], -1) - U[1][:])

        U_analytical = u(x, time * t, a, w, l, phi)

        error = np.mean(np.abs(U[1] - U_analytical))
        if error > max_error:
            max_error = error

        U[1] = U[0].copy()
        plt.pause(0.01)
    
    errors.append(max_error)

log_t = np.log(ts)
log_err = np.log(errors)

p, C = np.polyfit(log_t, log_err, 1)
print("Наклон (порядок сходимости):", p)

plt.loglog(ts, errors, marker='o', linestyle='-', color='r', label="Ошибка" )
plt.xlabel('Шаг по времени t')
plt.ylabel('Средняя ошибка')
plt.title('Проверка порядка сходимости')
plt.grid(True, which='both', ls='--', alpha=0.5)
plt.legend()
plt.show()