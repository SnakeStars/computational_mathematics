import numpy as np
import matplotlib.pyplot as plt

l = 20
a = 10
T = 1.0
skip_steps = 5

n_list = [51, 101, 201]
Emax_list = []

for n in n_list:
    h = 2 * l / n
    t = h / (2*a)
    steps = int(T / t) + 1

    U = np.zeros((n+2, n+2, 3))
    U[int(n/2)+1, int(n/2)+1, 1] = 1 / (h**2)

    U[1:-1,1:-1,0] = U[1:-1,1:-1,1] + ((t**2)*(a**2))/(2*(h**2)) * (
        U[2:,1:-1,1] + U[0:-2,1:-1,1] + U[1:-1,2:,1] + U[1:-1,0:-2,1] - 4*U[1:-1,1:-1,1])

    x = np.linspace(-l, l, n+2)
    y = np.linspace(-l, l, n+2)
    X, Y = np.meshgrid(x, y)

    time = t
    Emax_time = []
    for step in range(1, steps):

        U[1:-1,1:-1,2] = U[1:-1,1:-1,1]
        U[1:-1,1:-1,1] = U[1:-1,1:-1,0]
        U[1:-1,1:-1,0] = 2*U[1:-1,1:-1,1] - U[1:-1,1:-1,2] + ((a*t/h)**2) * (
            U[2:,1:-1,1] + U[0:-2,1:-1,1] + U[1:-1,2:,1] + U[1:-1,0:-2,1] - 4*U[1:-1,1:-1,1])

        time += t

        if step >= skip_steps:
            r = np.sqrt(X**2 + Y**2)
            U_analytic = np.zeros_like(U[:,:,0])
            mask = r < a * time
            U_analytic[mask] = 1 / (2*np.pi*a*np.sqrt((a*time)**2 - r[mask]**2))
            E = np.abs(U[:,:,0] - U_analytic)
            Emax_time.append(np.max(E))

    Emax_list.append(np.max(Emax_time))

h_values = [2*l/n for n in n_list]
Emax_array = np.array(Emax_list)

log_h = np.log(h_values)
log_E = np.log(Emax_array)

p, C = np.polyfit(log_h, log_E, 1)
print(f"Оценка порядка точности p ≈ {p:.2f}")

plt.figure(figsize=(6,4))
plt.plot(log_h, log_E, 'o')
plt.plot(log_h, p*log_h + C, 'r-', label=f'Наклон = {p:.2f}')
plt.xlabel('log(h)')
plt.ylabel('log(max error)')
plt.title('Оценка порядка точности')
plt.legend()
plt.grid(True)
plt.show()
