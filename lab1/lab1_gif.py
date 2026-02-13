import numpy as np
import matplotlib.pyplot as plt


l = 20
a = 10
n = 101
time = 0

T = l/a
h = 2*l/n
t = h/(2*a)

Ut = np.zeros((n, n))
U = np.zeros((n+2,n+2,3))
U[int(n/2) + 1][int(n/2) + 1][1] = 1/(h**2)
U[1:-1,1:-1,0] = U[1:-1,1:-1,1] + ((t**2) * (a**2))/(2 * (h**2)) * (
    U[2:,1:-1,1] + U[0:-2,1:-1,1] + U[1:-1,2:,1] + U[1:-1,0:-2,1] - 4*U[1:-1,1:-1,1])

x = np.linspace(-l, l, n+2)
y = np.linspace(-l, l, n+2)
X, Y = np.meshgrid(x, y)

Emax = []

fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(121, projection='3d')
surf = ax.plot_surface(X, Y, U[:,:,0], cmap='plasma')
ax.set_zlim(0, 1)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('u')
ax.set_title(f"t = {time:.3f}")

ax2 = fig.add_subplot(122)
ax2.set_xlabel("t")
ax2.set_ylabel("max error")
ax2.set_title("Максимальная ошибка")
line_Emax, = ax2.plot([], [], 'r-')
ax2.grid(True)

plt.ion()
plt.show()

while time < T:
    U[1:-1,1:-1,2] = U[1:-1,1:-1,1]
    U[1:-1,1:-1,1] = U[1:-1,1:-1,0]
    U[1:-1,1:-1,0] = 2 * U[1:-1,1:-1,1] - U[1:-1,1:-1,2] + ((a * t / h)**2) * (
        U[2:,1:-1,1] + U[0:-2,1:-1,1] + U[1:-1, 2:, 1] + U[1:-1, 0:-2, 1] - 4*U[1:-1,1:-1,1])
    
    r = np.sqrt(X**2 + Y**2)
    U_analytic = np.zeros_like(U[:,:,0])
    mask = r < a * time
    U_analytic[mask] = 1 / (2 * np.pi * a * np.sqrt((a*time)**2 - r[mask]**2))
    
    E = np.abs(U[:,:,0] - U_analytic)
    Emax.append(np.max(E))
    
    ax.clear()
    surf = ax.plot_surface(X, Y, U[:,:,0], cmap='plasma')
    ax.set_zlim(0, 1)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('u')
    ax.set_title(f"t = {time:.3f}")
    
    line_Emax.set_data(np.arange(len(Emax))*t, Emax)
    ax2.relim()
    ax2.autoscale_view()
    
    plt.pause(0.01)
    time += t

plt.ioff()
plt.show()