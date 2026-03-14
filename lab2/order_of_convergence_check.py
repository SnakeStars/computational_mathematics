import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import solve_banded
from scipy.special import erfc

n = 201
s = np.linspace(-5, 5, n)
nu = 0.8

a = 1
x = np.sinh(s)
Q = 2
k = 200
S = 0.01

t0 = 5
t_arr = [0.1, 0.05, 0.01]
errors = []
time = 0
tmax = 0
T = 3

func = np.zeros_like(x)
center = np.argmin(np.abs(x))
func[center] = Q * (1/(x[center + 1] - x[center]))

psi = lambda xi: 1/(np.sqrt(np.pi)*xi) * np.exp(-xi**2) - erfc(xi)

def u(x, t, Q, k, S, t0):
    x = np.abs(x)
    term1 = psi(x / (2*np.sqrt(t)))
    term2 = psi(x / (2*np.sqrt(t - t0))) if t > t0 else 0
    return Q*x/(2*k*S) * (term1 - term2)

U = np.zeros((2,n))

hplus = x[2:] - x[1:-1]
hminus = x[1:-1] - x[:-2]

for t in t_arr:
    A = -2 * a**2 * nu * t / ((hminus + hplus) * hminus)
    C = -2 * a**2 * nu * t / ((hminus + hplus) * hplus)
    B = 1 - A - C
    ab = np.zeros((3,n-2))
    ab[0,1:] = C[:-1]
    ab[1,:] = B
    ab[2,:-1] = A[1:]

    max_error = 0
    U[:] = 0.0
    tmax = int(T/t)

    for time in range(1,tmax):
        U[0][:] = U[1][:]
        u_xx_last = (2/(hplus + hminus)) * (((U[0][2:] - U[0][1:-1])/hplus) - ((U[0][1:-1] - U[0][:-2])/hminus))
        R = U[0][1:-1] + a**2 * (1 - nu) * t * u_xx_last + t * func[1:-1] * np.heaviside(t0 - time * t, 1)
        U[1][1:-1] = solve_banded((1,1), ab, R)

        mask = np.zeros_like(x, dtype=bool)
        mask[1:-1] = (np.abs(x[1:-1]) > 1e-4) & (np.abs(x[1:-1]) < 0.5)
        u_an = np.zeros_like(U[1])
        u_an[mask] = u(x[mask], time * t, Q, k, S, t0)
        error = np.mean(np.abs(U[1][mask] - u_an[mask]))
        max_error = error if error > max_error else max_error
    errors.append(max_error)

log_t = np.log(t_arr)
log_err = np.log(errors)

p, C = np.polyfit(log_t, log_err, 1)
print("Наклон (порядок сходимости):", p)

plt.loglog(t_arr, errors, marker='o', linestyle='-', color='r', label="Ошибка" )
plt.xlabel('Шаг по времени t')
plt.ylabel('Средняя ошибка')
plt.title('Проверка порядка сходимости')
plt.grid(True, which='both', ls='--', alpha=0.5)
plt.legend()
plt.show()