import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
COLORS = list(mcolors.TABLEAU_COLORS)


def simple_plot(pi, wn, sigma, w):
    fig, ax = plt.subplots(1, 3, figsize=[10, 5])

    ax[0].set_ylabel(r"$\Pi(i\omega_n)$")
    ax[0].set_xlabel(r"$\omega_n$")
    ax[0].plot(wn.T, pi.T, ".")

    ax[1].set_xlabel(r"$n$")
    ax[1].plot(pi.T, ".")

    ax[2].set_ylabel(r"$\sigma(\omega)$")
    ax[2].set_xlabel(r"$\omega$")
    ax[2].plot(w.T, sigma.T, ms=2, lw=1)

    plt.tight_layout()
    plt.show()


def infer_scales(Pi, sigma):
    N = len(Pi[0])
    M = len(sigma[0])
    norm = Pi[:, 0]
    PiN = Pi[:, -1]
    sum1 = 2 * np.sum(sigma, axis=-1) - np.take(sigma, 0, axis=-1)
    dm = np.pi * norm / sum1
    m = np.arange(M)
    sum2 = 2 * np.sum(m**2 * sigma, axis=-1)

    wmaxs = M * dm
    betas = 2 * N * np.sqrt((np.pi**3) * PiN / (dm**3 * sum2))
    return wmaxs, betas


def plot_basic(Pi, sigma, filename=None):
    fig, ax = plt.subplots(2, 2, figsize=[7, 5])
    ax[0, 0].set_ylabel(r"$\Pi_n$")
    plt.setp(ax[0, 0].get_xticklabels(), visible=False)
    ax[1, 0].set_ylabel(r"$\sqrt{n^2 \Pi_n}$")
    ax[1, 0].set_xlabel(r"$n$")
    ax[0, 1].set_ylabel(r"$\sigma_m$")
    plt.setp(ax[0, 1].get_xticklabels(), visible=False)
    ax[1, 1].set_ylabel(r"$\sqrt{ \sum_{r}^{n} n^2 \sigma_n }$")
    ax[1, 1].set_xlabel(r"$m$")

    N = len(Pi[0])
    n2Pi = np.sqrt(np.arange(N) ** 2 * Pi)
    for i in range(len(Pi)):
        ax[0, 0].plot(Pi[i], ".", c=COLORS[i % 10])
        ax[1, 0].plot(n2Pi[i], ".", c=COLORS[i % 10])
    M = len(sigma[0])
    cumul_sum2 = np.sqrt(np.cumsum(np.linspace(0, 1, M) ** 2 * sigma, axis=-1))
    for i in range(len(sigma)):
        ax[0, 1].plot(sigma[i], c=COLORS[i % 10])
        ax[1, 1].plot(cumul_sum2[i], c=COLORS[i % 10])

    fig.tight_layout()
    if filename is not None:
        fig.savefig(filename)
    plt.show()


def plot_scaled(Pi, sigma, betas, wmaxs, filename=None, default_wmax=20.0):
    fig, ax = plt.subplots(2, 2, figsize=[7, 5])
    ax[0, 0].set_ylabel(r"$\Pi(i\omega_n)$")
    plt.setp(ax[0, 0].get_xticklabels(), visible=False)
    ax[1, 0].set_ylabel(r"$\sqrt{\omega_n^2 \Pi(i\omega_n)}$")
    ax[1, 0].set_xlabel(r"$\omega_n$")
    ax[0, 1].set_ylabel(r"$\sigma(\omega)$")
    plt.setp(ax[0, 1].get_xticklabels(), visible=False)
    ax[1, 1].set_ylabel(r"$\sqrt{\int\frac{d\omega}{\pi}\omega^2\sigma(\omega)}$")
    ax[1, 1].set_xlabel(r"$\omega$")

    N = len(Pi[0])
    wn = (2 * np.pi / betas[:, np.newaxis]) * np.arange(N)
    n2Pi = np.sqrt(wn**2 * Pi)
    for i in range(len(Pi)):
        ax[0, 0].plot(
            wn[i], Pi[i], ".", c=COLORS[i % 10], markersize=5
        )
        ax[1, 0].plot(
            wn[i], n2Pi[i], ".", c=COLORS[i % 10], markersize=5
        )
    M = len(sigma[0])
    w = wmaxs[:, np.newaxis] * np.linspace(0, 1, M)
    cumul_sum2 = np.sqrt(np.cumsum(np.linspace(0, 1, M) ** 2 * sigma, axis=-1))
    for i in range(len(sigma)):
        ax[0, 1].plot(w[i], (default_wmax / wmaxs[i]) * sigma[i], c=COLORS[i % 10])
        ax[1, 1].plot(w[i], cumul_sum2[i], c=COLORS[i % 10])

    fig.tight_layout()
    if filename is not None:
        fig.savefig(filename)
    plt.show()


def plot_infer_scale(Pi, sigma, filename=None, default_wmax=20.0):
    wmaxs, betas = infer_scales(Pi, sigma)
    print(f" infered scales:\n  betas =\n{betas}\n  wmaxs =\n{wmaxs}")
    plot_scaled(Pi, sigma, betas, wmaxs, filename, default_wmax)
