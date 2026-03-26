import matplotlib.pyplot as plt
import numpy as np
from scipy.special import erfc

psi = lambda xi: 1/(np.sqrt(np.pi)*xi) * np.exp(-xi**2) - erfc(xi)

def u(x, t, Q, k, S, t0):
    x = np.abs(x)
    term1 = psi(x / (2*np.sqrt(t)))
    term2 = psi(x / (2*np.sqrt(t - t0))) if t > t0 else 0
    return Q*x/(2*k*S) * (term1 - term2)

n = 200
x = np.sinh(np.linspace(-5, 5, n))
Q = 2
k = 100
S = 0.01
t0 = 5
tmax = 10
time = 0

max_u = 0


plt.ion()
fig, ax = plt.subplots()
line, = ax.plot(x, u(x,1,Q,k,S,t0))
ax.set_xlim(-10, 10)
ax.set_ylim(-1, 4)

for time in range(1,tmax):
    u_cur = u(x, time, Q, k, S, t0)
    u_cur_max = np.max(u_cur)
    line.set_ydata(u_cur)
    max_u = u_cur_max if u_cur_max > max_u else max_u
    plt.pause(1)
print(max_u)