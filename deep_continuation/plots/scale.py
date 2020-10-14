#%%
import numpy as np
from scipy import integrate
from scipy.special import binom
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
COLORS = list(mcolors.TABLEAU_COLORS)


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


def log_reg_log_integral(integrand, N_reg=2048, N_log=1024, reg_max=10, log_pow=7):
    reg_smpl = np.linspace(-reg_max, reg_max, N_reg)
    log_smpl = np.logspace(np.log10(reg_max), log_pow, N_log)[1:]
    smpl_arr = np.concatenate([-np.flip(log_smpl), reg_smpl, log_smpl]) 
    return integrate.simps(integrand(smpl_arr), smpl_arr, axis=-1)


def pi_integral(spectral_function, beta, num_wn=20):
    omega_n = (2*np.pi/beta) * np.arange(0,num_wn)
    omega_n = omega_n[:, np.newaxis]
    integrand = lambda x: (1/np.pi) * x**2/(x**2+omega_n**2) * spectral_function(x)
    return log_reg_log_integral(integrand)


def second_moment(spectral_function):
    integrand = lambda x: (1/np.pi) * x**2 * spectral_function(x)
    return log_reg_log_integral(integrand)


def sigma1(x):
    c = np.array([-3,-2,0,2,3])*100
    w = np.array([1,0.6,0.2,0.6,1])*100
    h = np.array([0.1,0.25,0.3,0.25,0.1]) * np.pi
    m = np.array([1,9,1,9,1])
    n = np.array([0,7,0,2,0])
    return peak_sum(x, c, w, h, m, n)


N_wn = 20
fac=1.8


def sigma2(x): 
    return fac*sigma1(fac*x)


def sigma3(x): 
    return sigma1(x/fac)/fac


beta1 = 1
beta2 = fac*beta1
beta3 = beta1/fac

X = np.linspace(-1000,1000,1000)
S1 = sigma1(X)
S2 = sigma2(X)
S3 = sigma3(X)

W1 = (2*np.pi/beta1) * np.arange(0,N_wn)
W2 = (2*np.pi/beta2) * np.arange(0,N_wn)
W3 = (2*np.pi/beta3) * np.arange(0,N_wn)
P11 = pi_integral(sigma1, beta1, N_wn)
P22 = pi_integral(sigma2, beta2, N_wn)
P33 = pi_integral(sigma3, beta3, N_wn)
P21 = pi_integral(sigma2, beta1, N_wn)
P13 = pi_integral(sigma1, beta3, N_wn)
P12 = pi_integral(sigma1, beta2, N_wn)
P31 = pi_integral(sigma3, beta1, N_wn)


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






fig = plt.figure(figsize=[7.5,2.5], dpi=80)
ax1 = plt.subplot(131)
ax2 = plt.subplot(132)
ax3 = plt.subplot(133, sharey=ax2)
plt.setp(ax3.get_yticklabels(), visible=False)

ax1.plot(X, S2, linewidth=0.9, c=COLORS[1], label=r"$s\sigma(s\omega  )  $")
ax1.plot(X, S1, linewidth=1.5, c=COLORS[0], label=r"$ \sigma( \omega  )  $")
ax1.plot(X, S3, linewidth=0.9, c=COLORS[3], label=r"$ \sigma( \omega/s)/s$")
ax1.set_xlabel(r"$\omega$")
ax1.set_ylabel(r"$\sigma(\omega)$")
ax1.legend(handlelength=1, ncol=1, loc=(0.6,0.8), fontsize='small')

# ax1.text(0.03, 0.95, r'$\sigma(\omega)$', ha='left', va='top', transform=ax1.transAxes)

ax2.plot(W2, P12, '^', markersize=5,  c=COLORS[0], label=r"$ \sigma( \omega  )  ,s\beta  $")
ax2.plot(W1, P11, '.', markersize=9, c=COLORS[0], label=r"$ \sigma( \omega  )  , \beta  $")
ax2.plot(W3, P13, 'v', markersize=5,  c=COLORS[0], label=r"$ \sigma( \omega  )  , \beta/s$")
ax2.plot(W2, P22, '.', markersize=7, c=COLORS[1], label=r"$s\sigma(s\omega  )  ,s\beta  $")
ax2.plot(W1, P21, 'v', markersize=3,  c=COLORS[1], label=r"$s\sigma(s\omega  )  , \beta  $")
ax2.plot(W1, P31, '^', markersize=3,  c=COLORS[3], label=r"$ \sigma( \omega/s)/s, \beta  $")
ax2.plot(W3, P33, '.', markersize=3,  c=COLORS[3], label=r"$ \sigma( \omega/s)/s, \beta/s$")
# ax2.set_ylim(0,0.4)
ax2.set_xlabel(r"$\omega_n$")
ax2.set_ylabel(r"$\Pi(\omega_n)$")
# ax2.legend(handlelength=1)
# ax2.text(0.05, 0.95, r'$\Pi(i\omega_n)$', ha='left', va='top', transform=ax2.transAxes)

ax3.plot(P12, '^', markersize=5,  c=COLORS[0], label=r"$ \sigma( \omega  )  ,s\beta  $")
ax3.plot(P11, '.', markersize=9, c=COLORS[0], label=r"$ \sigma( \omega  )  , \beta  $")
ax3.plot(P13, 'v', markersize=5,  c=COLORS[0], label=r"$ \sigma( \omega  )  , \beta/s$")
ax3.plot(P22, '.', markersize=7, c=COLORS[1], label=r"$s\sigma(s\omega  )  ,s\beta  $")
ax3.plot(P21, 'v', markersize=3,  c=COLORS[1], label=r"$s\sigma(s\omega  )  , \beta  $")
ax3.plot(P31, '^', markersize=3,  c=COLORS[3], label=r"$ \sigma( \omega/s)/s, \beta  $")
ax3.plot(P33, '.', markersize=3,  c=COLORS[3], label=r"$ \sigma( \omega/s)/s, \beta/s$")
# ax3.set_ylim(0,0.4)
ax3.set_xlabel(r"$n$")
# ax3.legend(handlelength=1)
# ax2.text(0.05, 0.95, r'$\Pi(i\omega_n)$', ha='left', va='top', transform=ax2.transAxes)
ax3.legend(handlelength=1, ncol=1,loc=(-0.65,0.42), fontsize='small')

# plt.suptitle(r"$\frac{\langle\omega^2\rangle_1}{\langle\omega^2\rangle_2} = %6.5f \approx %6.5f = \left(\frac{\beta_2}{\beta_1}\right)^2$"%(second_moment(sigma1)/second_moment(sigma2), beta2**2/beta1**2), y=0.97)
# plt.show()
# plt.tight_layout()
plt.savefig("scale.pdf")
