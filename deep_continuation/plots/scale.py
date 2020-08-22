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


def pi_integral(num_wn, spectral_func, beta, wmax=10, N=2048, power=7):
    omega_n = (2*np.pi/beta) * np.arange(0,num_wn)
    center = np.linspace(-wmax,wmax,N)
    tail = np.logspace(np.log10(wmax), power, N//2)
    omega = np.concatenate([-np.flip(tail)[:-1], center, tail[1:]]) 
    w_grid, wn_grid = np.meshgrid(omega, omega_n)

    integrand = (1/np.pi) * w_grid**2 * spectral_func(w_grid) / (w_grid**2+wn_grid**2)
    integral = integrate.simps( integrand, w_grid, axis=-1)
    return integral


def tail_integral(spectral_func, wmax=10, N=2048, power=7):
    center = np.linspace(-wmax,wmax,N)
    tail = np.logspace(np.log10(wmax), power, N//2)
    w_grid = np.concatenate([-np.flip(tail)[:-1], center, tail[1:]]) 

    integrand = (1/np.pi) * w_grid**2 * spectral_func(w_grid)
    integral = integrate.simps( integrand, w_grid, axis=-1)
    return integral


def sigma1(x):
    c = np.array([-3,-2,0,2,3])
    w = np.array([1,0.6,0.2,0.6,1])
    h = np.array([0.15,0.25,0.3,0.25,0.15])
    m = np.array([1,9,1,9,1])
    n = np.array([0,7,0,2,0])
    return peak_sum(x, c, w, h, m, n)


N_wn = 20
fac=2

def sigma2(x): return fac*sigma1(fac*x)
def sigma3(x): return sigma1(x/fac)/fac

beta1 = 50
beta2 = fac*beta1
beta3 = beta1/fac


X = np.linspace(-5,5,1000)
S1 = sigma1(X)
S2 = sigma2(X)
S3 = sigma3(X)

P1 = pi_integral(N_wn, sigma1, beta1)
P2 = pi_integral(N_wn, sigma2, beta2)
P3 = pi_integral(N_wn, sigma3, beta3)

P4 = pi_integral(N_wn, sigma2, beta1)
P5 = pi_integral(N_wn, sigma1, beta3)
P6 = pi_integral(N_wn, sigma3, beta2)


W1 = (2*np.pi/beta1) * np.arange(0,N_wn)
W2 = (2*np.pi/beta2) * np.arange(0,N_wn)
W3 = (2*np.pi/beta3) * np.arange(0,N_wn)

# lw = 0.5
# from cycler import cycler
# plt.rcParams.update({
#     'figure.subplot.bottom': 0.15,
#     'figure.subplot.hspace': 0,
#     'figure.subplot.left': 0.05,
#     'figure.subplot.right': 0.99,
#     'figure.subplot.top': 0.8,
#     'figure.subplot.wspace': 0.1,
#     'axes.xmargin': 0,
#     'axes.ymargin': 0,
#     'axes.linewidth': lw,
#     'axes.prop_cycle': cycler('color', [
#         '#d62728',
#         '#1f77b4',
#         '#555555',
#         '#2ca02c',
#         '#9467bd',
#         '#ff7f0e',
#         '#8c564b',
#         '#e377c2',
#         '#7f7f7f',
#         '#bcbd22',
#         '#17becf'
#     ]),
#     'lines.linewidth': 2*lw,
#     'xtick.top': True,
#     'ytick.right': True,
#     'xtick.direction': 'in',
#     'ytick.direction': 'in',
#     'ytick.major.size': 2,
#     'xtick.major.size': 2,
#     'ytick.major.width': lw,
#     'xtick.major.width': lw,
#     'font.family': 'serif',
#     'font.serif': 'Times',
#     'font.size': 11.0,
#     'text.usetex': True,
#     'pgf.texsystem': "pdflatex",
#     'legend.handlelength': 1.0,
#     'legend.frameon': False,
#     'legend.borderpad': 0.3,
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

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=[10,3], dpi=80)
ax1.plot(X, S1, label=r"$\langle\omega^2\rangle_1 = %4.3f$"%tail_integral(sigma1))
ax1.plot(X, S2, label=r"$\langle\omega^2\rangle_2 = %4.3f$"%tail_integral(sigma2))
ax1.plot(X, S3, label=r"$\langle\omega^2\rangle_2 = %4.3f$"%tail_integral(sigma3))
ax1.set_xlabel(r"$\omega$")
ax1.legend(handlelength=1)
# ax1.text(0.03, 0.95, r'$\sigma(\omega)$', ha='left', va='top', transform=ax1.transAxes)

ax2.plot(P1, marker='.', markersize=12, linewidth=5, label=r"$\beta_1=%d$"%beta1)
ax2.plot(P2, marker='.', markersize=11, linewidth=4, label=r"$\beta_2=%d$"%beta2)
ax2.plot(P3, marker='.', markersize=10, linewidth=3, label=r"$\beta_3=%d$"%beta3)
ax2.plot(P4, marker='.', markersize=9, linewidth=2, label=r"$\beta_1=%d$"%beta1)
ax2.plot(P5, marker='.', markersize=8, linewidth=1, label=r"$\beta_2=%d$"%beta3)
ax2.plot(P6, marker='.', markersize=7, linewidth=0.5, label=r"$\beta_3=%d$"%beta2)
ax2.set_ylim(0,0.4)
ax2.set_xlabel(r"$n$")
ax2.legend(handlelength=1)
# ax2.text(0.05, 0.95, r'$\Pi(i\omega_n)$', ha='left', va='top', transform=ax2.transAxes)

ax3.plot(W1, P1, marker='.', markersize=12, linewidth=5, label=r"$\beta_1=%d$"%beta1)
ax3.plot(W2, P2, marker='.', markersize=11, linewidth=4, label=r"$\beta_2=%d$"%beta2)
ax3.plot(W3, P3, marker='.', markersize=10, linewidth=3, label=r"$\beta_3=%d$"%beta3)
ax3.plot(W1, P4, marker='.', markersize=9, linewidth=2, label=r"$\beta_1=%d$"%beta1)
ax3.plot(W3, P5, marker='.', markersize=8, linewidth=1, label=r"$\beta_2=%d$"%beta3)
ax3.plot(W2, P6, marker='.', markersize=7, linewidth=0.5, label=r"$\beta_3=%d$"%beta2)
ax3.set_ylim(0,0.4)
ax3.set_xlabel(r"$\omega_n$")
ax3.legend(handlelength=1)
# ax2.text(0.05, 0.95, r'$\Pi(i\omega_n)$', ha='left', va='top', transform=ax2.transAxes)


plt.suptitle(r"$\frac{\langle\omega^2\rangle_1}{\langle\omega^2\rangle_2} = %6.5f \approx %6.5f = \left(\frac{\beta_2}{\beta_1}\right)^2$"%(tail_integral(sigma1)/tail_integral(sigma2), beta2**2/beta1**2), y=0.97)
plt.show()
# plt.savefig("scale.pdf")
