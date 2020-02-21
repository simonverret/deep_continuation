#!/usr/bin/env python3
# %%
import numpy as np
from scipy.integrate import simps
from numpy.linalg import inv
import matplotlib.pyplot as plt

def fermi(w,T=15):
    return 1/(np.exp(w/T)+1)


def disp1(kx,ky,kz):
    return kx**2 + ky**2 + kz**2 + 2

def green1(kx,ky,kz,z): 
    return 1/(z-disp1(kx,ky,kz))

def Akw1(kx,ky,kz,w,eta): 
    return eta/((w-disp1(kx,ky,kz))**2 + eta**2)


def disp2(kx,ky,kz):
    return -(kx**2 + ky**2 + kz**2)

def green2(kx,ky,kz,z): 
    return 1/(z-disp2(kx,ky,kz))

def Akw2(kx,ky,kz,w,eta): 
    return eta/((w-disp2(kx,ky,kz))**2 + eta**2)


def dos(w, dim = 3, eta=0.1,
        kxGrd=np.linspace(0,3,100), 
        kyGrd=np.linspace(0,3,100), 
        kzGrd=np.linspace(0,3,100)
    ):

    # making the sparse mashgrid for integration
    _ = np.newaxis
    kx = kxGrd[_,_,_,:]
    ky = kyGrd[_,_,:,_] if dim>1 else 0
    kz = kzGrd[_,:,_,_] if dim>2 else 0
    w  =     w[:,_,_,_]

    A = Akw1(kx,ky,kz,w,eta) + Akw2(kx,ky,kz,w,eta)
    integral = np.sum( A , axis=(1,2,3)).squeeze()
    volume   = np.size(kx) * np.size(ky) * np.size(kz)
    return integral / volume


def pi0(w, dim=3, T=1/15, eta = 0.1,
        kxGrd=np.linspace(0,3,100), 
        kyGrd=np.linspace(0,3,100), 
        kzGrd=np.linspace(0,3,100), 
        wGrd =np.linspace(-10,10,100),
    ):

    # making the sparse meshgrid for integration
    _ = np.newaxis
    wp =  wGrd[_,_,_,_,:]
    kx = kxGrd[_,_,_,:,_]
    ky = kyGrd[_,_,:,_,_] if dim>1 else 0
    kz = kzGrd[_,:,_,_,_] if dim>2 else 0
    w  =     w[:,_,_,_,_]

    # integrand
    k2 = kx**2 + ky**2 + kz**2
    A1 = Akw1(kx,ky,kz,w,eta) + Akw2(kx,ky,kz,w,eta)
    A2 = Akw1(kx,ky,kz,w+wp,eta) + Akw2(kx,ky,kz,w+wp,eta)
    f1 = fermi(w, T)
    f2 = fermi(w+wp, T)

    integral = np.sum( -k2*A1*A2*(f1-f2)/(w+0.001) , axis=(1,2,3,4)).squeeze()
    volume   = np.size(kx) * np.size(ky) * np.size(kz) * np.size(wp)

    return integral/volume

w = np.linspace(-2,4,200)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=[4,4])
ax1.set_ylabel('dos')
ax2.set_ylabel('pi0')
ax2.set_xlabel('w')

ax1.plot(w, 0.1*dos(w,dim=1))
ax1.plot(w, 0.5*dos(w,dim=2))

ax2.plot(w, 0.1*pi0(w,dim=1))
ax2.plot(w, 0.5*pi0(w,dim=2))

# plt.plot(w, 1.0*dos(w,dim=3))
plt.show()


