#%%
import numpy as np
from scipy import integrate
from scipy.special import binom
import matplotlib.pyplot as plt


def gaussian(x, c, w, h):
    return (h/(np.sqrt(np.pi)*w))*np.exp(-((x-c)/w)**2)


def lorentzian(x, c, w, h):
    return (h/np.pi)*w/((x-c)**2+w**2)


def bernstein(x, m, n):
    return binom(m,n) * (x**n) * ((1-x)**(m-n)) * (x>=0) * (x<=1)


def free_bernstein(x, m, n, c=0, w=1, h=1):
    sq = np.sqrt(m+1)
    xx = (x-c)/(w*sq) + n/m
    return (h*sq/w)*bernstein(xx, m, n)


def peak(omega, center=0, width=1, height=1, type_m=0, type_n=0):
    out = 0
    out += (type_m == 0) * lorentzian(omega, center, width, height)
    out += (type_m == 1) * gaussian(omega, center, width, height)
    out += (type_m >= 2) * free_bernstein(omega, type_m, type_n, center, width, height)
    return out


def peak_sum(x, c, w, h, m, n):
    x = x[np.newaxis, :]
    while len(c.shape) < len(x.shape):
        c = c[:, np.newaxis]
        w = w[:, np.newaxis]
        h = h[:, np.newaxis]
        m = m[:, np.newaxis]
        n = n[:, np.newaxis]
    return peak(x, c, w, h, m, n).sum(axis=0)

def sigma(x):
    c = np.array([-4,-2,0,2,4])
    w = np.array([1,1,0.2,1,1])
    h = np.array([0.15,0.25,0.3,0.25,0.15])
    m = np.array([1,9,1,9,1])
    n = np.array([0,7,0,2,0])
    return peak_sum(x, c, w, h, m, n)

def pi_integral(num_wn, spectral_func, beta):
    omega_n = (2*np.pi/beta) * np.arange(0,num_wn)
    omega = np.linspace(-10,10,1000)
    w_grid, wn_grid = np.meshgrid(omega, omega_n)

    integrand = (1/np.pi) * w_grid**2 * spectral_func(w_grid) / (w_grid**2+wn_grid**2)
    integral = integrate.simps( integrand, w_grid, axis=-1)
    return integral

def tail_integral(spectral_func):
    w_grid = np.linspace(-10,10,1000)
    integrand = (1/np.pi) * w_grid**2 * spectral_func(w_grid)
    integral = integrate.simps( integrand, w_grid, axis=-1)
    return round(integral,3)

fac=3
def sigma2(x): return fac*sigma(fac*x)
N_wn = 20
beta = 10
beta2 = fac*beta

X = np.linspace(-5,5,1000)
S = sigma(X)
S2 = sigma2(X)

P = pi_integral(N_wn, sigma, beta)
P2 = pi_integral(N_wn, sigma2, beta2)


# lw = 0.3
# plt.rcParams.update({
#     'axes.xmargin': 0,
#     'axes.ymargin': 0,
#     'axes.linewidth': lw,
#     'xtick.top': True,
#     'ytick.right': True,
#     'xtick.direction': 'in',
#     'ytick.direction': 'in',
#     'ytick.major.size': 2,
#     'xtick.major.size': 2,
#     'ytick.major.width': lw,
#     'xtick.major.width': lw,
#     'font.family': 'serif',
#     # 'font.serif': 'cm'
#     'font.size': 11.0,
#     'text.usetex': True,
#     'pgf.texsystem': "pdflatex",
#     'legend.handlelength': 1.0,
#     'text.latex.preamble': [
#         # r'\usepackage[utf8]{inputenc}',
#         r'\usepackage[T1]{fontenc}',
#         # r'\usepackage{amsfonts}',
#         # r'\usepackage{amssymb}',
#         # r'\usepackage{amsmath}',
#         # r'\usepackage{esint}',
#         r'\usepackage{newtxmath}',
#         r'\usepackage{dsfont}',
#         r'\usepackage{bm}',
#         # r'\renewcommand\vec{\mathbf}',
#         r'\renewcommand\vec{\bm}',
#     ],
# })

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=[7,2], dpi=80)
ax1.plot(X,S, label=r"$\langle\omega^2\rangle=$"+f"{tail_integral(sigma)}")
ax1.plot(X,S2, label=r"$\langle\omega^2\rangle=$"+f"{tail_integral(sigma2)}")
ax1.set_xlabel(r"$\omega$")
ax1.legend()

ax2.plot(P, marker='.', label=r"$\beta=$"+f"{beta}")
ax2.plot(P2, marker='.', label=r"$\beta=$"+f"{beta2}")
ax2.set_xlabel(r"$n$")
ax2.legend()

plt.tight_layout()
plt.show()
