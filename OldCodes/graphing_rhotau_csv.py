# -*- coding: utf-8 -*-
"""
Created on Tue May 30 09:59:49 2023
@author: laksh
Creating plots similar to Figure 4
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from helper_funcs import *
import time
from pypet import Environment, cartesian_product, Trajectory
import logging
import os # For path names working under Linux and Windows
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings
from mpl_toolkits import mplot3d
import seaborn as sns

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)

def fluc_pred(x, sig_z2, B, N, K, rho, sig_eps2, r):
    dummy = (np.sqrt(((1 + x/sig_z2)**3)/(x / sig_z2 * (N/K)**2)) + B**2*sig_eps2/2/sig_z2)/r
    return dummy*sig_z2

def nofluc_pred(x, sig_z2, N, K, r):
    return np.sqrt(((1 + x/sig_z2)**3)/(x / sig_z2 * (N/K)**2 * r**2))*sig_z2

def plast_pred(x, sig_z2, B, N, K, rho, sig_eps2, r):
    dum1 = np.sqrt(((1 + x/sig_z2)**3)/(x/sig_z2*N**2/K**2))
    dum2 = ((1 - rho**2)*B**2*sig_eps2/2/sig_z2)
    return (dum1 + dum2)/r*sig_z2

def csv_sim_plot(fn, subfold):
    data = pd.read_csv(filepath_or_buffer = os.path.join("hdf5", subfold, fn))
    data['sig_s2'] = data['sig_s']**2
    data['sig_u2'] = data['sig_u']**2
    # data['sp1.sig_z2'] = data['sp1.Gaa'] + data['sp1.Gbb']*data.sig_eps**2 + data.sig_e**2
    # data['sp2.sig_z2'] = data['sp2.Gaa'] + data['sp2.Gbb']*data.sig_eps**2 + data.sig_e**2
    data['sig_z2'] = data['sp1.Gaa'] + data['sp1.Gbb']*data.sig_eps**2 + data.sig_e**2
    
    sig_z2 = np.unique(data.sig_z2)[0]
    N = 0.5
    K = 1
    rho = np.unique(data.rho)[0]
    r = np.unique(data.r)[0]
    sig_eps2 = np.unique(data.sig_eps**2)[0]
    B= np.unique(data['sp1.B'])
    
    x = np.arange(np.min(data.sig_u)**2, np.max(data.sig_u)**2, (np.max(data.sig_u)**2 - np.min(data.sig_u)**2)/100)
    fl = fluc_pred(x, sig_z2, B, N, K, rho, sig_eps2, r)
    nfl = nofluc_pred(x, sig_z2, N, K, r)
    pls = plast_pred(x, sig_z2, B, N, K, rho, sig_eps2, r)
    
    fig, ax = plt.subplots(figsize=(5,4))
    # plt.title(f"{fn}")
    plt.xlabel("Width of resource utilization function (sig_u^2)") #x label
    plt.ylabel("Width of stabilizing sel. function (sig_s^2)") #y label
    cmap = plt.get_cmap('gist_yarg')
    cm = plt.cm.get_cmap('gist_yarg')
    plt.plot(x, fl, c = 'blue', label = 'fluc. env.')
    plt.plot(x, pls, c = 'red', label = 'evol. plast.')
    plt.plot(x, nfl, c = 'green', label = 'no fluc.')
    im = plt.scatter(data.sig_u2, data.sig_s2, c = data.delz, s = 30, cmap = cm)
    plt.ylim((np.min(data.sig_s2), np.max(data.sig_s2)))
    plt.xlim((np.min(data.sig_u2), np.max(data.sig_u2)))
    clb = fig.colorbar(im, ax=ax)
    # clb.ax.set_clim(0.0, 10.0)
    clb.ax.set_ylabel('CD')
    plt.legend()
    plt.tight_layout()
    dumsav = 'images/' + subfold + '/' + fn[:-4] + '.png'
    # fig.savefig(dumsav)

# fn = 'noplast_rho_mid_tau_mid.csv'

# traje = 'dummy'


# Graphing Exploration of Tau and Rho #
vals = [0.1, 0.5, 0.9]
clues = ['low','mid','high']
plst_condition = ['noplast','evplast']
subfold = "Fin_log_exp_eps_rho_tau"

for i, rhodum in enumerate(vals):
    for j, taudum in enumerate(vals):
        for k, dumdum in enumerate(plst_condition):
            fn = f"{dumdum}_rho_{clues[i]}_tau_{clues[j]}.csv"
            traje = 'dummy'
            # print(f'STARTING {fn}')
            # csv_sim_plot(fn, subfold)
            
# Graphing Exploration of environmental variance sig_eps #
vals = [1, 5, 10, 30]
clues = ['vlow', 'low', 'mid', 'high']
vals = [5, 10]
clues = ['low', 'mid']
plst_condition = ['noplast','evplast']
for i, eps in enumerate(vals):
    for k, dumdum in enumerate(plst_condition):
        fn = f"{dumdum}_eps_{clues[i]}.csv"
        traje = 'dummy'
        # print(f'STARTING {fn}')
        # csv_sim_plot(fn, subfold)
        
fnl = ["Fluc_log.csv", "NoFluc_log.csv", "EvolvingPlasticity_log.csv"]
subfold = "dummy2"
for i in fnl:
    print(f'STARTING {i}')
    csv_sim_plot(i, subfold)

# plt.plot(x, nofluc_pred(x), c = 'red')
# plt.plot(x, fluc_pred(x), c = 'blue')



