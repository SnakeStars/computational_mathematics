import numpy as np, matplotlib.pyplot as plt
from scipy.linalg import solve_banded
from scipy.special import genlaguerre
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle


def computeVi(r_borders):
    return (1/3) * (r_borders[1:]**3 - r_borders[:-1]**3)

def computeRi(r_borders, volumes):
    return (0.25/volumes) * (r_borders[1:]**4 - r_borders[:-1]**4)

def computeQ(r_borders, a_0):
    return (-1/a_0) * (r_borders[1:]**2 - r_borders[:-1]**2)

def computeW(r_borders, r_centres):
    n = len(r_centres)
    w = np.zeros(n + 1)
    w[1:n] = r_borders[1:n]**2 / np.diff(r_centres)
    w[n] = r_borders[n]**2 / (r_borders[n] - r_centres[-1])
    return w

def buildA(w, Q):
    diag = w[:-1] + w[1:] + Q
    offdiag  = -w[1:-1]
    return diag, offdiag

def applyA(u, diag, off):
    res = diag * u
    res[:-1] += off * u[1:]
    res[1:]  += off * u[:-1]
    return res

def computeEigenValue(mu, diag, off, volumes, tol=1e-13, maxit=1000):
    n = len(volumes)

    ab = np.zeros((3, n))
    ab[0, 1:] = off
    ab[1, :] = diag - mu * volumes
    ab[2, :-1] = off

    u = np.random.default_rng(42).normal(size=n)
    u /= np.sqrt(volumes @ u**2)
    E_old = np.inf

    for i in range(1, maxit+1):
        u_next = solve_banded((1, 1), ab, volumes * u)
        u = u_next / np.sqrt(volumes @ u_next**2)
        E = u @ applyA(u, diag, off)
        if abs(E - E_old) < tol:
            break
        E_old = E
    
    if u[np.argmax(np.abs(u))] < 0:
        u = -u
        
    return E, u, i

def analyticPsi(r_centres, volumes, k, a_0):
    x = 2 * r_centres / (a_0 * k)
    psi = np.exp(-r_centres / (a_0 * k)) * genlaguerre(k - 1, 1)(x)
    return psi / np.sqrt(volumes @ psi**2)

def matchSign(u, psi_exact, volumes):
    return -u if volumes @ (u * psi_exact) < 0 else u

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
GRID    = dict(color="#d9d8d2", lw=0.8)
 
def _style(ax):
    ax.grid(True, **GRID); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color("#9a9992")
    ax.tick_params(colors="#52514e", labelsize=9)
 
def plotEigenFunctions(r_centres, U, P, E, fname="eigenfunctions.png"):
    fig, ax = plt.subplots(figsize=(11, 7))
    x_hi, tail = 80.0, r_centres > 1.0
    for k, (u, p, e) in enumerate(zip(U, P, E), start=1):
        c = PALETTE[k - 1]
        s = 0.40 / np.abs(u[tail]).max()
        band = Rectangle((0, k - 0.49), x_hi, 0.98,
                         transform=ax.transData, facecolor="none")
        ax.axhline(k, color="#d9d8d2", lw=0.8, xmin=0.06, zorder=1)
        for line in (ax.plot(r_centres, k + s * u, color=c, lw=2.0, zorder=3)[0],
                     ax.plot(r_centres, k + s * p, color="#52514e", lw=1.2,
                             ls=(0, (5, 3)), zorder=4)[0]):
            line.set_clip_path(band)
        ax.text(-2.0, k, f"k={k}", color=c, fontsize=11, fontweight="bold",
                ha="right", va="center")
        ax.text(x_hi + 1.5, k, f"E = {e:.5f}", color="#52514e", fontsize=9, va="center")
    ax.set_xlim(-8, 102); ax.set_ylim(0.3, 6.8)
    ax.set_yticks([]); ax.set_xticks(np.arange(0, 81, 10))
    ax.set_xlabel(r"$r\,/\,a_0$")
    ax.set_ylabel(r"$\psi_k(r)$, масштаб по амплитуде осцилляций")
    ax.set_title("Собственные функции: численные (сплошные) и аналитические (пунктир)",
                 fontsize=12, color="#0b0b0b", pad=12)
    ax.legend(handles=[Line2D([], [], color="#52514e", lw=2.0, label="численно"),
                       Line2D([], [], color="#52514e", lw=1.2, ls=(0, (5, 3)),
                              label="аналитически")],
              frameon=False, loc="lower right", fontsize=10)
    ax.grid(True, axis="x", **GRID); ax.set_axisbelow(True)
    for s_ in ("top", "right", "left"): ax.spines[s_].set_visible(False)
    ax.spines["bottom"].set_color("#9a9992")
    ax.tick_params(colors="#52514e", labelsize=9)
    fig.tight_layout(); fig.savefig(fname, dpi=160, facecolor="white"); plt.close(fig)
 
def plotEigenFunctionsGrid(r_centres, U, P, E, fname="eigenfunctions_grid.png"):
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for k, (ax, u, p, e) in enumerate(zip(axes.ravel(), U, P, E), start=1):
        c = PALETTE[k - 1]
        ax.plot(r_centres, u, color=c, lw=2.0, label="численно")
        ax.plot(r_centres, p, color="#52514e", lw=1.3, ls=(0, (5, 3)), label="аналитически")
        ax.axhline(0, color="#9a9992", lw=0.8)
        ax.set_xlim(0, min(r_centres[-1], 7 * k**2))
        ax.set_title(f"k = {k}:   E = {e:.6f}   (точно {-1/k**2:.6f})",
                     fontsize=10, color="#0b0b0b")
        ax.set_xlabel(r"$r\,/\,a_0$"); ax.set_ylabel(r"$\psi_k$")
        ax.legend(frameon=False, fontsize=9)
        _style(ax)
    fig.suptitle("Собственные функции по уровням", fontsize=13, y=0.99)
    fig.tight_layout(); fig.savefig(fname, dpi=160, facecolor="white"); plt.close(fig)
 
def plotProbabilityDensity(r_centres, U, a_0, fname="probability.png", levels=3):
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for k in range(1, levels + 1):
        p = U[k - 1]**2 * r_centres**2
        c = PALETTE[k - 1]
        ax.plot(r_centres, p, color=c, lw=2.0, zorder=3)
        j = np.argmax(p)
        ax.annotate(f"k={k}", (r_centres[j], p[j]), textcoords="offset points",
                    xytext=(6, 4), color=c, fontsize=10, fontweight="bold")
    r_max = r_centres[np.argmax(U[0]**2 * r_centres**2)]
    ax.axvline(r_max, color="#52514e", lw=1.0, ls=(0, (2, 3)), zorder=2)
    ax.annotate(f"максимум при $r$ = {r_max:.3f} $a_0$", (r_max, ax.get_ylim()[1]*0.78),
                xytext=(14, 0), textcoords="offset points", fontsize=10, color="#52514e")
    ax.set_xlim(0, 30); ax.set_xlabel(r"$r\,/\,a_0$")
    ax.set_ylabel(r"$p(r)=|\psi(r)|^2 r^2$")
    ax.set_title("Плотность вероятности для нижних уровней", fontsize=12, pad=12)
    _style(ax)
    fig.tight_layout(); fig.savefig(fname, dpi=160, facecolor="white"); plt.close(fig)

if __name__ == "__main__":
    a_0, r_0, n, n_levels = 1.0, 150.0, 6000, 6

    r_borders = np.linspace(0, r_0, n + 1)
    volumes = computeVi(r_borders)
    r_centres = computeRi(r_borders, volumes)
    Q = computeQ(r_borders, a_0)
    diag, off = buildA(computeW(r_borders, r_centres), Q)

    E_num, U, P = [], [], []

    header = f"{'k':>2} {'mu':>10} {'E числ':>12} {'E точн':>12} {'отн.ошибка':>12} {'итер':>5}"
    print(header)
    print("-" * len(header))
    for k in range(1, n_levels + 1):
        mu = -1.15 / k**2
        E, u, it = computeEigenValue(mu, diag, off, volumes)
        psi = analyticPsi(r_centres, volumes, k, a_0)
        u = matchSign(u, psi, volumes)
        E_num.append(E)
        U.append(u)
        P.append(psi)
        print(f"{k:>2} {mu:10.4f} {E:12.6f} {-1/k**2:12.6f} "
              f"{abs(E + 1/k**2)*k**2:12.2e} {it:>5}")

    #G = np.array([[U[i] @ (volumes * U[j]) for j in range(n_levels)]
    #              for i in range(n_levels)])
    #print(f"макс |(u_k,u_m)_B| при k≠m: {np.abs(G - np.eye(n_levels)).max():.2e}")
    r_max = r_centres[np.argmax(U[0]**2 * r_centres**2)]
    print(f"наиболее вероятное положение: r = {r_max:.4f} a_0")

    plotEigenFunctions(r_centres, U, P, E_num)
    plotEigenFunctionsGrid(r_centres, U, P, E_num)
    plotProbabilityDensity(r_centres, U, a_0)
