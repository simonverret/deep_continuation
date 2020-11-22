#%% modules
import numpy as np
from numpy.linalg import norm, pinv
import matplotlib.pyplot as plt


#%% Simple ill-conditionned matrix
def eps_matrix(eps, dim1, dim2):
    return (np.ones((dim1,dim2)) + np.vstack([eps*np.eye(dim2)]+[np.zeros(dim2) for i in range(dim1-dim2)]))

eps = 0.001
dim1 = 5
dim2 = 5
M = eps_matrix(eps, dim1, dim2)/dim2
print(M)
index1 = np.arange(dim1)
index2 = np.arange(dim2)
targets = np.random.randint(1,10,dim1)
processed = np.matmul(targets, M)

print(targets)
print(f"1/{dim2} *\n {M*dim2}")
print(processed)


# Examples of processed outputs pairs
fig, (ax1, ax2) = plt.subplots(1,2, figsize=(9,1))
ax1.bar(index1, targets, width=0.8)
ax2.bar(index2, processed, width=0.8)
ax1.set_xticks([])
ax2.set_xticks([])
ax1.set_ylim(0,10)
ax2.set_ylim(0,10)
plt.show()


# Solution exacte par inversion
Minv = pinv(M)
amps = np.array([0.00001, 0.00005, 0.0001, 0.0005, 0.001])
avg_error = np.zeros_like(amps)

fig, ax = plt.subplots(len(amps),2, figsize=(8,8))
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

for i, amp in enumerate(amps):
    noise = np.random.normal(0, amp, dim2)
    outputs = np.matmul(processed + noise , Minv)
    
    ax[i,0].bar(index2,processed+noise, width=0.4, color=colors[3], linewidth=2)
    # Make some labels.
    rects = ax[i,0].patches
    for rect, label in zip(rects, noise):
        posx = rect.get_x()+rect.get_width()/2
        posy = rect.get_height()
        ax[i,0].text(posx, posy, f"{label:+1.0e}", ha='center', va='bottom', rotation=45)
    ax[i,0].bar(index2,processed, width=0.8, color=colors[3], linewidth=5, alpha=0.5)
    ax[i,1].bar(index1, outputs, width=0.4, color=colors[3], linewidth=2)
    ax[i,1].bar(index1, targets, width=0.8, color=colors[3], linewidth=5, alpha=0.5)

    ax[i,0].set_title(f"input & noise ($\sigma=${amp:0.5f})")
    ax[i,0].set_ylim(0,10)
    ax[i,0].set_xticks([])
    
    ax[i,1].set_title("output")
    ax[i,1].set_ylim(0,10)
    ax[i,1].set_yticks([])
    ax[i,1].set_xticks([])

plt.tight_layout()
plt.show()


#%% condition
def vec_norm_broadcast(arr, ord=2):
    return norm(arr, axis=-1, ord=ord)

def mat_norm_broadcast(arr, ord=2):
    return norm(arr, axis=(-2,-1), ord=ord)

def cond_number(M):
    norm_Minv =  mat_norm_broadcast(pinv(M))
    norm_M =  mat_norm_broadcast(M)
    return norm_M * norm_Minv

eps = 0.0001
dim1 = 5
dim2 = 5
M = eps_matrix(eps, dim1, dim2)/dim2
Minv = pinv(M)

kappa = cond_number(M)
print(f"condition number = {kappa}")

batch = 100
targets = np.random.randint(1,10, (batch, dim1))
processed = np.matmul(targets, M)

amps = np.linspace(1e-5,1e-2,100)
err_abs_in = np.zeros((len(amps),batch))
err_abs_out = np.zeros((len(amps),batch))
err_abs_ratio = np.zeros((len(amps),batch))
err_rel_out = np.zeros((len(amps),batch))
err_rel_in = np.zeros((len(amps),batch))
err_rel_ratio = np.zeros((len(amps),batch))

for i, amp in enumerate(amps):
    noise = np.random.normal(0,amp, (batch,dim2))
    outputs = np.matmul(processed + noise , Minv)
    
    err_abs_in[i] = vec_norm_broadcast(noise)
    err_abs_out[i] = vec_norm_broadcast(outputs-targets)
    err_abs_ratio[i] = err_abs_out[i]/err_abs_in[i]
    err_rel_in[i] = err_abs_in[i]/vec_norm_broadcast(processed)
    err_rel_out[i] = err_abs_out[i]/vec_norm_broadcast(targets)
    err_rel_ratio[i] = err_rel_out[i]/err_rel_in[i]


plt.plot(err_rel_in, err_rel_out, '.', color=colors[0])
# plt.plot(np.mean(err_rel_in), np.mean(err_rel_out, axis=-1))
# plt.plot(np.mean(err_rel_in), np.min(err_rel_out, axis=-1))

# plt.plot(np.mean(err_rel_in), kappa * np.mean(err_rel_in, axis=-1))
x = np.linspace(0,np.max(err_rel_in))
plt.plot(x, kappa * x, color=colors[3])

plt.title("error out vs error in")
plt.xlabel("relative error in")
plt.ylabel("relative error out")
maxx = 0.001
plt.xlim(0,maxx)
plt.ylim(0,kappa*maxx)

plt.show()


#%%
eps = np.logspace(-5,5,100)
for dim in [2,8,32,128,512]:
    kap = cond_number(eps_matrix(eps[:, np.newaxis, np.newaxis], dim, dim))
    plt.plot(eps, kap, label=dim)
    plt.xscale('log')
    plt.yscale('log')
plt.legend(title="dimension")
plt.xlabel("$\epsilon$ (log scale)")
plt.ylabel("condition number $\kappa$ (log scale)")
plt.show()


#%% Condition number of various analytic continuation kernel

def matsub(wmax=10, Nw=101, beta=10, Nt=101, domain="frequency", stats="bosons"):
    if domain in ["time","t"]:
        tmin = 0
        tmax = beta
    elif domain in ["frequency","freq","w"]:
        if stats in ["bosons","boson","bose","b"]:
            tmin = 0
            tmax = (Nt-1)*2*np.pi/beta
        elif stats in ["fermions","fermion","fermi","f"]:
            tmin = np.pi/beta
            tmax = (Nt-1)*2*np.pi/beta
        else: raise ValueError(f"stats {stats} not recognized")
    else: raise ValueError(f"domain {domain} not recognized")
    w = np.linspace(-wmax, wmax, Nw, axis=-1)
    t = np.linspace(tmin, tmax, Nt, axis=-1)
    return w, t

def kernel_matrix(kernel_func, w, t):
    w = np.expand_dims(w, axis=-2)
    t = np.expand_dims(t, axis=-1)
    return np.nan_to_num( kernel_func(w, t), copy=False, nan=1)

def plot_kernel(
        title, kernel_func,
        wmax=5, Nw=201, beta=5, Nt=51, domain="frequency", stats="bosons"
    ):
    w, t = matsub(wmax, Nw, beta, Nt, domain, stats)
    if domain in ["time","t"]:
        kfunc = lambda w, t: kernel_func(w, t, beta)
    else:
        kfunc = kernel_func

    kM = kernel_matrix(kfunc, w, t)
    colors = plt.cm.coolwarm(np.linspace(0,1,len(kM)))
    for i,k in enumerate(kM):
        plt.plot(w,k, color=colors[i])
    plt.xlabel(r"$\omega$")
    plt.title(title)
    tmap = plt.cm.ScalarMappable(
        cmap=plt.cm.coolwarm,
        norm=plt.Normalize(vmin=t.min(), vmax=t.max())
    )
    cbar = plt.colorbar(tmap)
    if domain in ["time","t"]:
        cbar.ax.set_ylabel(r"$\tau$", rotation=270, labelpad=15)
        plt.ylabel(r"$K(\omega,\tau)$")
    else:
        cbar.ax.set_ylabel(r"$\omega_n$", rotation=270, labelpad=15)
        plt.ylabel(r"$K(\omega,\omega_n)$")
    plt.show()


def fbph_kernel(w, wn):
    return w**2/(w**2+wn**2)
plot_kernel(
    r"$K(\omega,\omega_n) = \dfrac{\omega^2}{\omega^2 + \omega_n^2}$",
    fbph_kernel, domain="freq", stats="bose"
)

#%%
def ffph_kernel(w, wn):
    return -wn/(w**2+wn**2)
plot_kernel(
    r"$K(\omega,\omega_n) = \dfrac{-\omega_n}{\omega^2 + \omega_n^2}$",
    ffph_kernel, domain="freq", stats="fermi"
)

#%%
def tf_kernel(w, t, beta):
    return -np.exp(-w*t)/(1+np.exp(-w*beta))
plot_kernel(
    r"$K(\omega,\tau) = \dfrac{-e^{-\omega\tau}}{1 + e^{-\omega\beta}}$",
    tf_kernel, domain="time", stats="fermions"
)

#%%
def tb_kernel(w, t, beta):
    w = w+1e-9
    return (w/2)*(np.exp(-w*t) + np.exp(-w*(beta-t)))/(1-np.exp(-w*beta))
plot_kernel(
    r"$K(\omega,\tau) = \dfrac{\omega}{2}\dfrac{e^{-\omega\tau}+e^{-\omega(\beta-\tau)}}{1 - e^{-\omega\beta}}$",
    tb_kernel, domain="time", stats="bose",
)


#%% would require real and imaginary parts
# def ff_kernel(w, wn):
#     return 1/(1j*wn-w)

# def fb_kernel(w, wn):
#     return w/(w+1j*wn)



#%% example spectrum and matsubara
from deep_continuation.data_generator import free_beta, sum_on_args
from scipy import integrate

c = np.array([-5,-3,0,3,5])
w = np.array([1,0.6,0.2,0.6,1])
h = np.array([0.1,0.25,0.3,0.25,0.1]) * np.pi
a = np.array([20,7,10,2,20])
b = np.array([20,2,10,7,20])
sigma_func = lambda x: sum_on_args(free_beta, x, c, w, h, a, b)

def integrate_with_tails(integrand, grid_points=4096, tail_points=1024, grid_end=10, tail_power=7):
    grid_sampling = np.linspace(-grid_end, grid_end, grid_points)
    # tail_sampling = np.logspace(np.log10(grid_end), tail_power, tail_points)[1:]
    full_sampling = grid_sampling
    # np.concatenate([
    #     -np.flip(tail_sampling),
    #     grid_sampling,
    #     tail_sampling
    # ])
    return (1/np.pi)*integrate.simps(integrand(full_sampling), full_sampling, axis=-1)
    # return np.sum(integrand(full_sampling), axis=-1)


def matsub_integral(t, kernel_func, spectral_function, **kwargs):
    if isinstance(t, np.ndarray):
        t = t[:, np.newaxis]    
    def integrand(w): return kernel_func(w,t) * spectral_function(w)
    return integrate_with_tails(integrand, **kwargs)


from cycler import cycler

def plot_spectrum(
        domain="frequency", stats="bosons",
        wmax=10, Nw=201, beta=10, Nt=101,
    ):
    w, t = matsub(wmax, Nw, beta, Nt, domain, stats)

    fig, ax = plt.subplots(1, 2, figsize=(10,2))
    ax[0].plot(w, sigma_func(w), 'k')

    if domain in ["time","t"]:
        if stats in ["bosons","boson","bose","b"]:
            kfunc = lambda w, t: tb_kernel(w, t, beta)
            ax[0].set_ylabel(r'$\chi''(\omega)$')
            ax[1].set_ylabel(r'$\chi(\tau)$')
        elif stats in ["fermions","fermion","fermi","f"]:
            kfunc = lambda w, t: tf_kernel(w, t, beta)
            ax[0].set_ylabel(r'$A(\omega)$')
            ax[1].set_ylabel(r'$G(\tau)$')
        else: raise ValueError(f"stats {stats} not recognized")
        ax[0].set_xlabel(r'$\omega$')
        ax[1].set_xlabel(r'$\tau$')
        # ax[1].plot(t, matsub_integral(t, kfunc, sigma_func), '-k')

        cm = plt.get_cmap('coolwarm')
        yy =  matsub_integral(t, kfunc, sigma_func)
        ax[1].set_prop_cycle(cycler('color',[cm(1.*i/(len(t)-1)) for i in range(len(t)-1)]))
        for jj in range(len(t)):
            ax[1].plot(t[jj],yy[jj],'.')

    elif domain in ["frequency","freq","w"]:
        if stats in ["bosons","boson","bose","b"]:
            kfunc = fbph_kernel
            ax[0].set_ylabel(r'$\chi''(\omega)/\omega$')
            ax[1].set_ylabel(r'$\chi(i\omega_n)$')
        elif stats in ["fermions","fermion","fermi","f"]:
            kfunc = ffph_kernel
            ax[0].set_ylabel(r'$A(\omega)$')
            ax[1].set_ylabel(r'$G(i\omega_n)$')
        else: raise ValueError(f"stats {stats} not recognized")
        ax[0].set_xlabel(r'$\omega$')
        ax[1].set_xlabel(r'$\omega_n$')
        # ax[1].plot(t, matsub_integral(t, kfunc, sigma_func), '.k')

        cm = plt.get_cmap('coolwarm')
        yy =  matsub_integral(t, kfunc, sigma_func)
        ax[1].set_prop_cycle(cycler('color',[cm(1.*i/(len(t)-1)) for i in range(len(t)-1)]))
        for jj in range(len(t)):
            ax[1].plot(t[jj],yy[jj],'.')

    else: raise ValueError(f"domain {domain} not recognized")
    plt.tight_layout()
    plt.show()


plot_spectrum('w','b')
plot_spectrum('w','f')
plot_spectrum('t','f')
plot_spectrum('t','b')


#%%

def matsub(wmax=10, Nw=101, beta=10, Nt=101, domain="frequency", stats="bosons"):
    if domain in ["time","t"]:
        tmin = 0
        tmax = beta
    elif domain in ["frequency","freq","w"]:
        if stats in ["bosons","boson","bose","b"]:
            tmin = 0
            tmax = (Nt-1)*2*np.pi/beta
        elif stats in ["fermions","fermion","fermi","f"]:
            tmin = np.pi/beta
            tmax = (Nt-1)*2*np.pi/beta
        else: raise ValueError(f"stats {stats} not recognized")
    else: raise ValueError(f"domain {domain} not recognized")
    w = np.linspace(-wmax, wmax, Nw, axis=-1)
    t = np.linspace(tmin, tmax, Nt, axis=-1)
    return w, t


def plot_condition_number(
        domain="frequency", stats="bosons", N = 64,
        wmax_arr = np.logspace(0,3,400)[np.newaxis,:],
        beta_arr = np.logspace(0,2,3)[:,np.newaxis],
    ):
    if domain in ["time","t"]:
        beta_arr4 = beta_arr[:, :, np.newaxis, np.newaxis]
        if stats in ["bosons","boson","bose","b"]:
            kfunc = lambda w, t: tb_kernel(w, t, beta_arr4)
        elif stats in ["fermions","fermion","fermi","f"]:
            kfunc = lambda w, t: tf_kernel(w, t, beta_arr4)
        else: raise ValueError(f"stats {stats} not recognized")
    elif domain in ["frequency","freq","w"]:
        if stats in ["bosons","boson","bose","b"]:
            kfunc = fbph_kernel
        elif stats in ["fermions","fermion","fermi","f"]:
            kfunc = ffph_kernel
        else: raise ValueError(f"stats {stats} not recognized")
    else: raise ValueError(f"domain {domain} not recognized")

    w, t = matsub(wmax_arr, N, beta_arr, N, domain='t', stats='f')
    kernel_matrices = kernel_matrix(kfunc, w, t)
    kap = cond_number(kernel_matrices)

    fig, ax = plt.subplots(len(beta_arr),1, figsize=(6,5))
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for i, (b, k) in enumerate(zip(beta_arr, kap)):
        ax[i].plot(wmax_arr[0], k, color = colors[3])
        ax[i].set_title(f"beta = {b[0]}")
        ax[i].set_xscale('log')
        ax[i].set_yscale('log')
        ax[i].set_ylabel("$\kappa$")
        if i != len(beta_arr)-1:
            plt.setp(ax[i].get_xticklabels(), visible=False)
        else:
            ax[i].set_xlabel("wmax")
    plt.tight_layout()
    plt.show()

plot_condition_number('w','b')
plot_condition_number('w','f')
plot_condition_number('t','b')
plot_condition_number('t','f')
