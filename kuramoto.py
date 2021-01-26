import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.io import savemat
# KSequ.m - solution of Kuramoto-Sivashinsky equation
#
# u_t = -u*u_x - u_xx - u_xxxx, periodic boundary conditions on [0,32*pi]
# computation is based on v = fft(u), so linear term is diagonal
#
# Using this program:
# u is the initial condition
# h is the time step
# N is the number of points calculated along x
# a is the max value in the initial condition
# b is the min value in the initial condition
# x is used when using a periodic boundary condition, to set up in terms of
#   pi
#
# Initial condition and grid setup
N = 50
#x = 32.0*2.0*np.pi*np.transpose(np.conj(np.arange(1, N+1))) / N
x = 16.75*2*np.transpose(np.conj(np.arange(1, N+1))) / N

lenng=np.max(x)-np.min(x)

x=x-np.min(x)

a = -lenng/2
b = lenng/2

amin = -lenng/2
bmax = lenng/2

u = np.cos(x*2*np.pi/(b-a))#*(1+np.sin(x/16))
#u=np.exp(-1.0*abs((x-lenng/2)**2.0))
v = np.fft.fft(u)
# scalars for ETDRK4
h = 0.005/1
#k = np.transpose(np.conj(np.concatenate((np.arange(0, N/2), np.array([0]), np.arange(-N/2+1, 0))))) / 16

k = np.transpose(np.conj(np.concatenate((np.arange(0, N/2), np.array([0]), np.arange(-N/2+1, 0))))) * 2*np.pi/(bmax-amin)

L = k**2 - k**4
E = np.exp(h*L)
E_2 = np.exp(h*L/2)
M = 16
r = np.exp(1j*np.pi*(np.arange(1, M+1)-0.5) / M)
LR = h*np.transpose(np.repeat([L], M, axis=0)) + np.repeat([r], N, axis=0)
Q = h*np.real(np.mean((np.exp(LR/2)-1)/LR, axis=1))
f1 = h*np.real(np.mean((-4-LR+np.exp(LR)*(4-3*LR+LR**2))/LR**3, axis=1))
f2 = h*np.real(np.mean((2+LR+np.exp(LR)*(-2+LR))/LR**3, axis=1))
f3 = h*np.real(np.mean((-4-3*LR-LR**2+np.exp(LR)*(4-LR))/LR**3, axis=1))
# main loop
uu = np.array([u])
tt = 0
tmax = 0.500
nmax = round(tmax/h)

nplt = int((tmax/100)/h)
g = -1*0.25j*k


#g = -1j*k

saveur=np.zeros([N,nmax],dtype=float)
saveui=np.zeros([N,nmax],dtype=float)
saveux=np.zeros([N,nmax],dtype=float)
saveuxx=np.zeros([N,nmax],dtype=float)
for n in range(1, nmax+1):
    t = n*h
    Nv = g*np.fft.fft(np.real(np.fft.ifft(v))**2)
    a = E_2*v + Q*Nv
    Na = g*np.fft.fft(np.real(np.fft.ifft(a))**2)
    b = E_2*v + Q*Na
    Nb = g*np.fft.fft(np.real(np.fft.ifft(b))**2)
    c = E_2*a + Q*(2*Nb-Nv)
    Nc = g*np.fft.fft(np.real(np.fft.ifft(c))**2)
    v = E*v + Nv*f1 + 2*(Na+Nb)*f2 + Nc*f3
    u = np.real(np.fft.ifft(v))
    
    
    saveur[:,n-1]=np.real(u[:])
    saveui[:,n-1]=np.imag(u[:])
    
    du=-1j*k/1.0*np.fft.fft(u)
    du=np.fft.ifft(du)
    
    ddu=(-1j*k/1.0)*(-1j*k/1.0)*np.fft.fft(u)
    ddu=np.fft.ifft(ddu)
    
    saveux[:,n-1]=np.real(du[:])
    saveuxx[:,n-1]=np.real(ddu[:])
    
    
#    if n%nplt == 0:
#        u = np.real(np.fft.ifft(v))
#        uu = np.append(uu, np.array([u]), axis=0)
#        tt = np.hstack((tt, t))
# plot
#fig = plt.figure()
#ax = fig.gca(projection='3d')
#tt, x = np.meshgrid(tt, x)
#surf = ax.plot_surface(tt, x, uu.transpose(), cmap=cm.coolwarm, linewidth=0, antialiased=False)
#fig.colorbar(surf, shrink=0.5, aspect=5)
#plt.show()
kkkk=np.zeros([N,1],dtype=float)
x1=np.zeros([1,N],dtype=float)
usol=np.zeros([1,N],dtype=float)
uinit=np.zeros([1,N],dtype=float)
for I in range(0,N):
    usol[0,I]=u[I]
    kkkk[I,0]=k[I]
    x1[0,I]=0+64*np.pi/(N-1)*I
    uinit[0,I] = np.cos(x1[0,I]/16)#*(1+np.sin(x1[0,I]/16))
    
fig = plt.figure()
plt.scatter(x1[0,:],np.real(usol[0,:]), color='black', linewidth=3, label='Spectral solution')
plt.legend()
plt.show()



fig = plt.figure()
plt.scatter(x1[0,:],np.real(saveuxx[:,n-1]), color='black', linewidth=3, label='Spectral solution')
plt.legend()
plt.show()









#Amp=0.01/2
#Amp=0.01/16
Amp=0.000
    
saveusquared=np.zeros([N,nmax],dtype=float)
for I in range(0,nmax):
    noise1 = Amp*np.random.normal(0,1,N)
    noise2 = Amp*np.random.normal(0,1,N)
    noise3 = Amp*np.random.normal(0,1,N)
    noise4 = Amp*np.random.normal(0,1,N)
    for K in range(0,N):
        saveur[K,I]=saveur[K,I]+noise1[K]
        saveui[K,I]=saveui[K,I]+noise2[K]
#        saveusquared[K,I]=np.sqrt(saveur[K,I]**2.0+saveui[K,I]**2.0)
        saveux[K,I]=np.real(saveux[K,I])+noise3[K]
        saveuxx[K,I]=np.real(saveuxx[K,I])+noise4[K]






#savemat('dataNLS.mat',{'x':x1, 'ur':saveur,'ui':saveui, 'usquared':saveusquared, 'dt':dt,'a':a,'b':b})

#savemat('datakuramoto.mat',{'x':x1, 'ur':saveur,'ui':saveui, 'usquared':saveusquared, 'dt':dt,'a':a,'b':b})
savemat('datakuramoto.mat',{'x':x, 'ur':saveur,'ui':saveui, 'deex':saveux, 'deexx':saveuxx, 'dt':h,'a':amin,'b':bmax})












#fig = plt.figure()
#plt.scatter(x1[0,:],np.real(uinit[0,:]), color='black', linewidth=3, label='Spectral solution')
#plt.legend()
#plt.show()

#Amp=0.000    
    
#for I in range(0,Nt):
#    noise1 = Amp*np.random.normal(0,1,Nx)
#    noise2 = Amp*np.random.normal(0,1,Nx)
#    for K in range(0,Nx):
#        rsaveu[K,I]=rsaveu[K,I]+noise1[K]
#        rsaveu[K,I]=rsaveu[K,I]+noise2[K]

#savemat('dataspectral.mat',{'x':x1, 'ur':rsaveu,'ui':isaveu, 'dt':dt,'a':a,'b':b})
