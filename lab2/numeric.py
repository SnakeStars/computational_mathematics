import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import solve_banded
from scipy.special import erfc

psi = lambda xi: 1/(np.sqrt(np.pi)*xi) * np.exp(-xi**2) - erfc(xi)

def u_analytical(x, t, Q, k, S, t0):
    x = np.abs(x)
    term1 = psi(x / (2*np.sqrt(t)))
    term2 = psi(x / (2*np.sqrt(t - t0))) if t > t0 else 0
    return Q*x/(2*k*S) * (term1 - term2)

n = 201
s = np.linspace(-5, 5, n)
nu = 0.5

a = 1
x = np.sinh(s)
Q = 2000
k = 100
S = 0.01

t0 = 5
t = 0.01
time = 0
T = 10

func = np.zeros_like(x)
center = np.argmin(np.abs(x))
func[center] = Q * (1/(x[center + 1] - x[center]))

U = np.zeros((2,n))
plt.ion()
fig, ax = plt.subplots()
line, = ax.plot(x, U[1])
line_analytical, = ax.plot(x, np.zeros_like(x),color='red', linestyle='--')
ax.set_xlim(-10, 10)
ax.set_ylim(-1, Q*4)

hplus = x[2:] - x[1:-1]
hminus = x[1:-1] - x[:-2]

A = -2 * a**2 * nu * t / ((hminus + hplus) * hminus)
C = -2 * a**2 * nu * t / ((hminus + hplus) * hplus)
B = 1 - A - C
ab = np.zeros((3,n-2))
ab[0,1:] = C[:-1]
ab[1,:] = B
ab[2,:-1] = A[1:]

max_u = 0

tmax = int(T/t)

for time in range(1,tmax):
    U[0][:] = U[1][:]
    u_xx_last = (2/(hplus + hminus)) * (((U[0][2:] - U[0][1:-1])/hplus) - ((U[0][1:-1] - U[0][:-2])/hminus))
    R = U[0][1:-1] + a**2 * (1 - nu) * t * u_xx_last + t * func[1:-1] * np.heaviside(t0 - time * t, 1)
    U[1][1:-1] = solve_banded((1,1), ab, R)
    line.set_ydata(U[1])
    line_analytical.set_ydata(u_analytical(x, t*time, Q, k, S, t0))
    u_cur_max = np.max(U[1])
    max_u = u_cur_max if u_cur_max > max_u else max_u
    plt.pause(0.01)
print(max_u)