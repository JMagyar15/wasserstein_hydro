import numpy as np
from Wasserstein import hydro_models as hm
import matplotlib.pyplot as plt
from scipy.optimize import Bounds

N = 100 #number of grid points in each model dimension
true_sol = np.array([3,5]) #true model parameters

np.random.seed(seed=452118) #reproducibility of results


t = np.linspace(0,500,200)
r,r_n = hm.Simple_Rain(200,10)
UH = hm.HydroModel(t,hm.Unit_Hydrograph)

obs = UH(true_sol,r) #observed streamflow to be fitted


RMSE = np.zeros([N,N]) #misfit surface for RMSE
W2_100 = np.zeros([N,N]) #misfit surface for W2
W2_1000 = np.zeros([N,N]) #misfit surface for W2
W2_10000 = np.zeros([N,N]) #misfit surface for W2

RMSE_opt = np.zeros([N,N]) #optimised RMSE for each initialisation
W2_100opt = np.zeros([N,N]) #optimised W2 for each initialisation
W2_1000opt = np.zeros([N,N]) #optimised W2 for each initialisation
W2_10000opt = np.zeros([N,N]) #optimised W2 for each initialisation

#model norm error of optimised solutions
RMSE_err = np.zeros([N,N]) 
W2_100err = np.zeros([N,N])
W2_1000err = np.zeros([N,N])
W2_10000err = np.zeros([N,N])

#initialisation points
theta_test = np.linspace(0.5,15.1,N) 
k_test = np.linspace(1.5,12.1,N)

#lower bounds on parameters for optimisation algorithm
bounds = Bounds(np.array([1e-5,1+1e-5]),np.array([16,13]),keep_feasible=True)

print('Set up completed, optimisation over grid...')

for i,theta in enumerate(theta_test):
    for j,k in enumerate(k_test):
        sim = UH([theta,k],r)
        W2_100[i,j] = hm.Wasserstein(obs,sim,gamma=100)
        W2_1000[i,j] = hm.Wasserstein(obs,sim,gamma=1000)
        W2_10000[i,j] = hm.Wasserstein(obs,sim,gamma=10000)

        RMSE[i,j] = np.sqrt(np.sum((sim.q - obs.q)**2)/sim.N)
        
        RMSE_sol = hm.Fit_Hydrograph(obs,r,UH,np.array([theta,k]),misfit='RMSE',method='Nelder-Mead',bounds=bounds)
        W2_100sol = hm.Fit_Hydrograph(obs,r,UH,np.array([theta,k]),misfit='W2',method='Nelder-Mead',gamma=100,bounds=bounds)
        W2_1000sol = hm.Fit_Hydrograph(obs,r,UH,np.array([theta,k]),misfit='W2',method='Nelder-Mead',gamma=1000,bounds=bounds)
        W2_10000sol = hm.Fit_Hydrograph(obs,r,UH,np.array([theta,k]),misfit='W2',method='Nelder-Mead',gamma=10000,bounds=bounds)

        RMSE_opt[i,j] = RMSE_sol.fun
        W2_100opt[i,j] = W2_100sol.fun
        W2_1000opt[i,j] = W2_1000sol.fun
        W2_10000opt[i,j] = W2_10000sol.fun

        RMSE_err[i,j] = np.linalg.norm((RMSE_sol.x - true_sol))
        W2_100err[i,j] = np.linalg.norm((W2_100sol.x - true_sol))
        W2_1000err[i,j] = np.linalg.norm((W2_1000sol.x - true_sol))
        W2_10000err[i,j] = np.linalg.norm((W2_10000sol.x - true_sol))

print('Optimisation complete, plotting underway...')


fig, ax = plt.subplots(ncols=2,nrows=4,figsize=(8,9),sharex=True,sharey=True)
cRMSE = ax[0,0].contourf(theta_test,k_test,RMSE.T,cmap='Blues',levels=30)
cW2_100 = ax[1,0].contourf(theta_test,k_test,np.sqrt(W2_100).T,cmap='Blues',levels=30)
cW2_1000 = ax[2,0].contourf(theta_test,k_test,np.sqrt(W2_1000).T,cmap='Blues',levels=30)
cW2_10000 = ax[3,0].contourf(theta_test,k_test,np.sqrt(W2_10000).T,cmap='Blues',levels=30)

cbRMSE = plt.colorbar(cRMSE,ax=ax[0,0])
cbW2_100 = plt.colorbar(cW2_100,ax=ax[1,0])
cbW2_1000 = plt.colorbar(cW2_1000,ax=ax[2,0])
cbW2_10000 = plt.colorbar(cW2_10000,ax=ax[3,0])


for i in range(4):
    ax[i,0].plot(3,5,'or',label='Truth')
    ax[i,0].legend(fontsize=9)
    ax[i,1].set_xlim((theta_test[0],theta_test[-1]))
    ax[i,1].set_ylim((k_test[0],k_test[-1]))
    ax[i,0].set_ylabel(r'$k$')


ax[3,0].set_xlabel(r'$\theta$')
ax[3,1].set_xlabel(r'$\theta$')


radius = 1e-3

THETA, K = np.meshgrid(theta_test,k_test)

ax[0,1].scatter(THETA[RMSE_err<radius],K[RMSE_err<radius],color='blue',marker='x',s=1)
ax[0,1].scatter(THETA[RMSE_err>=radius],K[RMSE_err>=radius],color='red',marker='x',s=1)

ax[1,1].scatter(THETA[W2_100err<radius],K[W2_100err<radius],color='blue',marker='x',s=1)
ax[1,1].scatter(THETA[W2_100err>=radius],K[W2_100err>=radius],color='red',marker='x',s=1)

ax[2,1].scatter(THETA[W2_1000err<radius],K[W2_1000err<radius],color='blue',marker='x',s=1)
ax[2,1].scatter(THETA[W2_1000err>=radius],K[W2_1000err>=radius],color='red',marker='x',s=1)

ax[3,1].scatter(THETA[W2_10000err<radius],K[W2_10000err<radius],color='blue',marker='x',s=1)
ax[3,1].scatter(THETA[W2_10000err>=radius],K[W2_10000err>=radius],color='red',marker='x',s=1)

cbRMSE.set_label('RMSE')
cbW2_100.set_label(r'$W_{2,100}$')
cbW2_1000.set_label(r'$W_{2,1000}$')
cbW2_10000.set_label(r'$W_{2,10000}$')

plt.subplots_adjust(wspace=0.35,hspace=0.4)

plt.savefig('Figures/iuh_misfit.pdf',bbox_inches='tight',dpi=400)
plt.savefig('Figures/iuh_misfit.png',bbox_inches='tight',dpi=400)

print('Completed!')