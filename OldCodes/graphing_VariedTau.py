# -*- coding: utf-8 -*-
"""
Created on Tue May 30 09:59:49 2023
@author: laksh
Graphing Figure 6 of report
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

# Graphing Exploration of Assymetric conditions#

def plotter_taus(data, xax, yax, xlab, ylab, tit):
    plt.rcParams.update({'font.size': 9})
    fig, ax = plt.subplots(figsize=(5,4))
    for i,n in enumerate(taus):
        dumdat = data[data['tau1'] == n]
        plt.plot(dumdat[xax], dumdat[yax], label=n)
    plt.legend(title = "tau1", fontsize=8)
    # ax.set_ylim(4,10)
    plt.xlabel(f"{xlab}", fontsize=10)
    plt.ylabel(f"{ylab}", fontsize=10)
    plt.tight_layout()
    # plt.title(f"{tit}")

subfold = "Fin_log_VaryingTau"
fn = "Larger_TauVar.csv"
fn = "Larger_TauVar_fin10k_sigs1000.csv"

data = pd.read_csv(filepath_or_buffer = os.path.join("hdf5", subfold, fn))
taus = np.sort(np.unique(data['tau1']))

# plotting simple delb vs tau2#
xlab = "tau2"
ylab = "divergence in plasticity: b1 - b2"
tit = "delb vs tau2 for varying tau1"
data['delb'] = data['sp1.b'] - data['sp2.b']
plotter_taus(data, 'tau2', 'delb', xlab, ylab, tit)

xlab = "tau2"
ylab = "CD in plasticity"
tit = "delb vs tau2 for varying tau1"
data['delb'] = (data['sp1.b'] - data['sp2.b'])/(data['sp1.B']*data['rho'] - data['sp2.B']*data['rho']**(data['tau2']/data['tau1']))
plotter_taus(data, 'tau2', 'delb', xlab, ylab, tit)

# plotting simple dela vs tau2#
xlab = "tau2"
ylab = "CD in trait: a1 - a2"
tit = "dela vs tau2 for varying tau1"
plotter_taus(data, 'tau2', 'dela', xlab, ylab, tit)

# plotting simple delz vs tau2 #
xlab = "tau2"
ylab = "CD in expresssed trait: z1 - z2"
tit = "delz vs tau2 for varying tau1"
plotter_taus(data, 'tau2', 'delz', xlab, ylab, tit)

# plotting population of 1 vs tau2 #
xlab = "tau2"
ylab = "Final population of species 1"
tit = "Pop sp1 vs tau2 for varying tau1"
plotter_taus(data, 'tau2', 'sp1.n', xlab, ylab, tit)

# plotting population of 2 vs tau2 #
xlab = "tau2"
ylab = "Final population of species 2"
tit = "Pop sp2 vs tau2 for varying tau1"
plotter_taus(data, 'tau2', 'sp2.n', xlab, ylab, tit)

data['b1_del_thet'] = data['sp1.b'] - data['sp1.B']*data['rho']
# plotting b1 - optima vs tau2 #
xlab = "tau2"
ylab = "b1 - B*rho"
tit = "b1 - B*rho vs tau2 for varying tau1"
plotter_taus(data, 'tau2', 'b1_del_thet', xlab, ylab, tit)

data['b2_del_thet'] = data['sp2.b'] - data['sp2.B']*data['rho']**(data['tau2']/data['tau1'])
# plotting b1 - optima vs b0 of 2 #
xlab = "tau2"
ylab = "b2 - B*rho'"
tit = "b2 - B*rho' vs tau2 for varying tau1"
plotter_taus(data, 'tau2', 'b2_del_thet', xlab, ylab, tit)

print(fn)

# NEED TO TRY SMOOTHING!!!