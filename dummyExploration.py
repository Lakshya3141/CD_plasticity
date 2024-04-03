import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)


def fluc_pred_new(x, sig_z2, B, sig_eps2, r):
    return (4*x + 5*sig_z2 + (B**2)*sig_eps2)/(2*r)


def nofluc_pred_new(x, sig_z2, r):
    return (4*x + 5*sig_z2)/(2*r)

def plast_pred_new(x, sig_z2, B, rho, sig_eps2, r):
    return (4*x + 5*sig_z2 + (B**2)*sig_eps2*(1-rho**2))/(2*r)


fnl = ["NoFluc_main.csv" , "Fluc_main.csv", "EvolvingPlasticity_main.csv"]
subfold = "final_sim"

datas = []
for i, fn in enumerate(fnl):
    datas.append(pd.read_csv(filepath_or_buffer=os.path.join("hdf5", subfold, fn)))
    datas[i]['sig_s2'] = datas[i]['sig_s']**2
    datas[i]['sig_u2'] = datas[i]['sig_u']**2
    datas[i]['sig_z2'] = datas[i]['sp1.Gaa'] + datas[i]['sp1.Gbb']*datas[i].sig_eps**2 + datas[i].sig_e**2

sig_z2 = np.unique(datas[0].sig_z2)[0]
rho = np.unique(datas[0].rho)[0]
r = np.unique(datas[0].r)[0]
sig_eps2 = np.unique(datas[0].sig_eps**2)[0]
B = np.unique(datas[0]['sp1.B'])

x = np.arange(np.min(datas[0].sig_u)**2, np.max(datas[0].sig_u)**2, (np.max(datas[0].sig_u)**2 - np.min(datas[0].sig_u)**2)/100)
nfl = nofluc_pred_new(x, sig_z2, r)
fl = fluc_pred_new(x, sig_z2, B, sig_eps2, r)
pls = plast_pred_new(x, sig_z2, B, rho, sig_eps2, r)

fig, ax = plt.subplots(figsize=(12,8))
plt.xlabel("Width of resource utilization function ($\sigma_u^2$)", fontsize=22) 
plt.ylabel("Width of stabilizing selection function ($\sigma_s^2$)", fontsize=22)  

plt.ylim((np.min(datas[0].sig_s2), np.max(datas[0].sig_s2)))
plt.xlim((np.min(datas[0].sig_u2), np.max(datas[0].sig_u2)))

plt.plot(x, nfl, c = 'tab:orange', label = 'No fluctuations', linewidth=3, alpha=1.0)
plt.plot(x, fl, c = 'tab:blue', label = 'Fluctuating environment', linewidth=3, alpha=1.0)
plt.plot(x, pls, c = 'tab:green', label = 'Evolving plasticity', linewidth=3, alpha=1.0)

transp = 0.5
size = 35
xnoise = 0.01
ynoise = 0.01
# For datas[0], colormap goes from void to orange
cm0 = plt.cm.get_cmap('Oranges').copy()
cm0.set_under('none')  # Set color under the lowest value to 'none'
im0 = plt.scatter(datas[0].sig_u2 + np.random.normal(scale=xnoise, size=len(datas[0])), datas[0].sig_s2 + np.random.normal(scale=ynoise, size=len(datas[0])), c=datas[0].delz, s=size, cmap=cm0, alpha=1.0)

# For datas[2], colormap goes from void to green
cm2 = plt.cm.get_cmap('Greens').copy()
cm2.set_under('none')  # Set color under the lowest value to 'none'
im2 = plt.scatter(datas[2].sig_u2 + np.random.normal(scale=xnoise, size=len(datas[2])), datas[2].sig_s2 + np.random.normal(scale=ynoise, size=len(datas[2])), c=datas[2].delz, s=size, cmap=cm2, alpha=0.8)

# For datas[1], colormap goes from void to blue
cm1 = plt.cm.get_cmap('Blues').copy()
cm1.set_under('none')  # Set color under the lowest value to 'none'
im1 = plt.scatter(datas[1].sig_u2 + np.random.normal(scale=xnoise, size=len(datas[1])), datas[1].sig_s2 + np.random.normal(scale=ynoise, size=len(datas[1])), c=datas[1].delz, s=size, cmap=cm1, alpha=0.4)

clb2 = fig.colorbar(im2, ax=ax, ticks=np.arange(0, 13, 2), pad=0.02, fraction=0.1,  shrink=0.8) # Adjust shrink value
clb2.ax.set_ylabel('CD : |$z̅_1$ - $z̅_2$|', fontsize=22)
clb2.ax.tick_params(labelsize=19)

clb1 = fig.colorbar(im1, ax=ax, ticks=np.arange(0, 13, 2), pad=0.02, fraction=0.085, shrink=0.8) # Adjust shrink value
# clb1.ax.set_ylabel('CD : |$z̅_1$ - $z̅_2$|', fontsize=22)
clb1.ax.tick_params(labelsize=19)


clb0 = fig.colorbar(im0, ax=ax, ticks=np.arange(0, 13, 2), pad=0.02, fraction=0.1, shrink=0.8) # Adjust shrink value
# clb0.ax.set_ylabel('CD : |$z̅_1$ - $z̅_2$|', fontsize=22)
clb0.ax.tick_params(labelsize=19)


plt.legend(loc='lower right', fontsize=19)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.tight_layout()

plt.show()
