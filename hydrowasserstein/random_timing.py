import numpy as np
from Wasserstein import hydro_models as hm
import matplotlib.pyplot as plt
from scipy.optimize import Bounds

np.random.seed(seed=46326623) #make this reproducible

true = [10,2,0.7]

bounds = Bounds(np.array([5,1,0.1]),np.array([15,4,1]),keep_feasible=True)
n_events = 500
t = np.linspace(0,200,200)
NL = hm.HydroModel(t,hm.NonLinearHydro)

W2_1_loc = []
W2_2_loc = []
RMSE_loc = []
HW_loc = []

for n in range(n_events):
    r,r_n = hm.Simple_Rain(200,5)
    
    obs = NL([10,2,0.7],r)
    
    W2_1_loc.append(hm.Fit_Hydrograph(obs,r_n,NL,np.array([10,2,0.7]),misfit='W2',method='Nelder-Mead',gamma=10).x)
    W2_2_loc.append(hm.Fit_Hydrograph(obs,r_n,NL,np.array([10,2,0.7]),misfit='W2',method='Nelder-Mead',gamma=100).x)
    RMSE_loc.append(hm.Fit_Hydrograph(obs,r_n,NL,np.array([10,2,0.7]),misfit='RMSE',method='Nelder-Mead').x)
    HW_loc.append(hm.Fit_Hydrograph(obs,r_n,NL,np.array([10,2,0.7]),misfit='HW',method='Nelder-Mead').x)


RMSE_loc = np.array(RMSE_loc)
W2_1_loc = np.array(W2_1_loc)
W2_2_loc = np.array(W2_2_loc)
HW_loc = np.array(HW_loc)


fig, ax = plt.subplots(ncols=2,nrows=2,figsize=(8,8))

ax[0,0].plot(RMSE_loc[:,0],RMSE_loc[:,1],'xk')
ax[0,0].set_title('RMSE')

ax[0,1].plot(W2_1_loc[:,0],W2_1_loc[:,1],'xk')
ax[0,1].set_title(r'$W_{2,10}$')

ax[1,0].plot(W2_2_loc[:,0],W2_2_loc[:,1],'xk')
ax[1,0].set_title(r'$W_{2,100}$')

ax[1,1].plot(HW_loc[:,0],HW_loc[:,1],'xk')
ax[1,1].set_title(r'$HW_2$')


for i in range(4):
    ax[i//2,i%2].set_xlim((5,15))
    ax[i//2,i%2].set_ylim((1,4))
    ax[i//2,i%2].plot(true[0],true[1],'or')
    ax[i//2,i%2].set_xlabel(r'$m_1$')
    ax[i//2,i%2].set_ylabel(r'$m_2$')
    
plt.subplots_adjust(hspace=0.3,wspace=0.3)

plt.savefig('Figures/end12.pdf',bbox_inches='tight')
plt.savefig('Figures/end12.png',bbox_inches='tight')

fig, ax = plt.subplots(ncols=2,nrows=2,figsize=(8,8))

ax[0,0].plot(RMSE_loc[:,0],RMSE_loc[:,2],'xk')
ax[0,0].set_title('RMSE')

ax[0,1].plot(W2_1_loc[:,0],W2_1_loc[:,2],'xk')
ax[0,1].set_title(r'$W_{2,10}$')

ax[1,0].plot(W2_2_loc[:,0],W2_2_loc[:,2],'xk')
ax[1,0].set_title(r'$W_{2,100}$')

ax[1,1].plot(HW_loc[:,0],HW_loc[:,2],'xk')
ax[1,1].set_title(r'$HW_2$')

for i in range(4):
    ax[i//2,i%2].set_xlim((5,15))
    ax[i//2,i%2].set_ylim((0.1,1))
    ax[i//2,i%2].plot(true[0],true[2],'or')
    ax[i//2,i%2].set_xlabel(r'$m_1$')
    ax[i//2,i%2].set_ylabel(r'$m_3$')
    
plt.subplots_adjust(hspace=0.3,wspace=0.3)

plt.savefig('Figures/end13.pdf',bbox_inches='tight')
plt.savefig('Figures/end13.png',bbox_inches='tight')

fig, ax = plt.subplots(ncols=2,nrows=2,figsize=(8,8))

ax[0,0].plot(RMSE_loc[:,1],RMSE_loc[:,2],'xk')
ax[0,0].set_title('RMSE')

ax[0,1].plot(W2_1_loc[:,1],W2_1_loc[:,2],'xk')
ax[0,1].set_title(r'$W_{2,10}$')

ax[1,0].plot(W2_2_loc[:,1],W2_2_loc[:,2],'xk')
ax[1,0].set_title(r'$W_{2,100}$')

ax[1,1].plot(HW_loc[:,1],HW_loc[:,2],'xk')
ax[1,1].set_title(r'$HW_2$')

for i in range(4):
    ax[i//2,i%2].set_xlim((1,4))
    ax[i//2,i%2].set_ylim((0.1,1))
    ax[i//2,i%2].plot(true[1],true[2],'or')
    ax[i//2,i%2].set_xlabel(r'$m_2$')
    ax[i//2,i%2].set_ylabel(r'$m_3$')
    
plt.subplots_adjust(hspace=0.3,wspace=0.3)

plt.savefig('Figures/end23.pdf',bbox_inches='tight')
plt.savefig('Figures/end23.png',bbox_inches='tight')

fig, ax = plt.subplots(ncols=3,figsize=(10,6))

ax[0].axhline(10,xmin=0,xmax=1,color='darkblue',ls='-.')
ax[1].axhline(2,xmin=0,xmax=1,color='darkblue',ls='-.')
ax[2].axhline(0.7,xmin=0,xmax=1,color='darkblue',ls='-.')

medianprops = dict(color='red',lw=1.5)

ax[0].boxplot([RMSE_loc[:,0],W2_1_loc[:,0],W2_2_loc[:,0],HW_loc[:,0]],labels=('RMSE',r'$W_{2,10}$',r'$W_{2,100}$',r'$HW_2$'),medianprops=medianprops)
ax[1].boxplot([RMSE_loc[:,1],W2_1_loc[:,1],W2_2_loc[:,1],HW_loc[:,1]],labels=('RMSE',r'$W_{2,10}$',r'$W_{2,100}$',r'$HW_2$'),medianprops=medianprops)
ax[2].boxplot([RMSE_loc[:,2],W2_1_loc[:,2],W2_2_loc[:,2],HW_loc[:,2]],labels=('RMSE',r'$W_{2,10}$',r'$W_{2,100}$',r'$HW_2$'),medianprops=medianprops)
ax[0].set_title(r'$m_1$')
ax[1].set_title(r'$m_2$')
ax[2].set_title(r'$m_3$')

ax[0].set_ylim((3,25))
ax[1].set_ylim((1,5))
ax[2].set_ylim((0,1))

plt.savefig('Figures/boxplot.pdf',bbox_inches='tight',dpi=400)
plt.savefig('Figures/boxplot.png',bbox_inches='tight',dpi=400)