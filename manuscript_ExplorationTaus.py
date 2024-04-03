import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)


def fluc_pred_new(x, sig_z2, B, sig_eps2, r):
    return (4*x + 5*sig_z2 + (B**2)*sig_eps2)/(2*r)


def nofluc_pred_new(x, sig_z2, r):
    return (4*x + 5*sig_z2)/(2*r)

def plast_pred_new(x, sig_z2, B, rho, sig_eps2, r):
    return (4*x + 5*sig_z2 + (B**2)*sig_eps2*(1-rho**2))/(2*r)


fnl = ["lowDiff.csv" , "MidDiff.csv", "HighDiff.csv"]
subfold = "tau_exp_large"

# Create 'images' folder if it doesn't exist
if not os.path.exists(os.path.join('images', subfold)):
    os.makedirs(os.path.join('images', subfold))

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

leg_size = 23


for i in range(0,3):
    print(i)
    
    ## Plotting the evolving scenario
    fig, ax = plt.subplots(figsize=(10,8))
    plt.xlabel("Resource utilization breadth ($\sigma_u^2$)", fontsize=28) 
    plt.ylabel("Stabilizing selection breadth ($\sigma_s^2$)", fontsize=28)  
    plt.ylim((np.min(datas[i].sig_s2), np.max(datas[i].sig_s2)))
    plt.xlim((np.min(datas[i].sig_u2), np.max(datas[i].sig_u2)))
    cm0 = plt.cm.get_cmap('gist_yarg')
    im0 = plt.scatter(datas[i].sig_u2, datas[i].sig_s2, c=datas[i].delz, s=40, cmap=cm0, alpha=1.0)
    # plt.plot(x, nfl, c = 'tab:orange', label = 'No fluctuations', linewidth=3, alpha=1.0)
    # plt.plot(x, fl, c = 'tab:blue', label = 'Fluctuating env.', linewidth=3, alpha=1.0)
    # plt.plot(x, pls, c = 'tab:green', label = 'Evolving plasticity', linewidth=3, alpha=1.0)
    clb0 = fig.colorbar(im0, ax=ax, ticks=np.arange(0, 13, 0.4), pad=0.02, fraction=0.1, shrink=0.8) # Adjust shrink value
    clb0.ax.set_ylabel('CD : |$z̅_1$ - $z̅_2$|', fontsize=25)
    clb0.ax.tick_params(labelsize=18)
    ax.tick_params(axis='both', which='major', labelsize=23)
    # plt.legend(loc='lower right', fontsize=leg_size)
    plt.tight_layout()
    plt.show()
    dumsav = 'images/' + subfold + '/' + fnl[i][:-4] + '.jpg'
    fig.savefig(dumsav, dpi=550)
    
    
    ## Plasticity plotter
    fig, ax = plt.subplots(figsize=(10,8))
    plt.xlabel("Resource utilization breadth ($\sigma_u^2$)", fontsize=28) 
    plt.ylabel("Stabilizing selection breadth ($\sigma_s^2$)", fontsize=28)  
    plt.ylim((np.min(datas[i].sig_s2), np.max(datas[i].sig_s2)))
    plt.xlim((np.min(datas[i].sig_u2), np.max(datas[i].sig_u2)))
    cm0 = plt.cm.get_cmap('gist_yarg')
    im0 = plt.scatter(datas[i].sig_u2, datas[i].sig_s2, c=datas[i].delb, s=40, cmap=cm0, alpha=1.0)
    plt.plot(x, nfl, c = 'tab:orange', label = 'No fluctuations', linewidth=3, alpha=1.0)
    plt.plot(x, fl, c = 'tab:blue', label = 'Fluctuating env.', linewidth=3, alpha=1.0)
    plt.plot(x, pls, c = 'tab:green', label = 'Evolving plasticity', linewidth=3, alpha=1.0)
    clb0 = fig.colorbar(im0, ax=ax, ticks=np.arange(0, 13, 0.3), pad=0.02, fraction=0.1, shrink=0.8) # Adjust shrink value
    clb0.ax.set_ylabel('CD : |$b̅_1$ - $b̅_2$|', fontsize=25)
    clb0.ax.tick_params(labelsize=18)
    ax.tick_params(axis='both', which='major', labelsize=23)
    plt.legend(loc='lower right', fontsize=leg_size)
    plt.tight_layout()
    plt.show()
    dumsav = 'images/' + subfold + '/' + fnl[i][:-4] + "_delb" + '.jpg'
    fig.savefig(dumsav, dpi=550)