# -*- coding: utf-8 -*-
"""
Created on Tue May 30 09:59:49 2023

@author: laksh
graphing Figure 5 of report
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

def plotter_pops(data, xax, yax, xlab, ylab, tit):
    plt.rcParams.update({'font.size': 9})
    fig, ax = plt.subplots(figsize=(5,4))
    for i,n in enumerate(pops):
        dumdat = data[data['sp2.n0'] == n]
        plt.plot(dumdat[xax], dumdat[yax], label=n)
    plt.legend(title = "Fixed pop 2", fontsize=8)
    plt.xlabel(f"{xlab}", fontsize=10)
    plt.ylabel(f"{ylab}", fontsize=10)
    # plt.title(f"{tit}")

subfold = "Fin_log_CpopCplast"
fn = "CDisp_Sp2CpCpop.csv"

data = pd.read_csv(filepath_or_buffer = os.path.join("hdf5", subfold, fn))
pops = np.sort(np.unique(data['sp2.n0']))[:-2]

# plotting simple delb vs b0 of 2 #
xlab = "Fixed plasticity of species 2"
ylab = "CD in plasticity: b1 - b2"
tit = "delb vs plast sp2 for varying constant n2"
data['delb'] = data['sp1.b'] - data['sp2.b']
plotter_pops(data, 'sp2.b0', 'delb', xlab, ylab, tit)

# plotting simple dela vs b0 of 2 #
# xlab = "Plasticity of species 2"
ylab = "CD in trait: a1 - a2"
tit = "dela vs plast sp2 for varying constant n2"
plotter_pops(data, 'sp2.b0', 'dela', xlab, ylab, tit)

# plotting simple delz vs b0 of 2 #
# xlab = "Plasticity of species 2"
ylab = "CD in expressed trait: z1 - z2"
tit = "delz vs plast sp2 for varying constant n2"
plotter_pops(data, 'sp2.b0', 'delz', xlab, ylab, tit)

# plotting population of 1 vs b0 of 2 #
# xlab = "Plasticity of species 2"
ylab = "Final stationary population of species 1"
tit = "Pop sp1 vs plast sp2 for varying constant n2"
plotter_pops(data, 'sp2.b0', 'sp1.n', xlab, ylab, tit)

data['b1_del_thet'] = data['sp1.b'] - data['sp1.B']*data['rho']
# plotting b1 - optima vs b0 of 2 #
# xlab = "Plasticity of species 2"
ylab = "b1 - B*rho"
tit = "b1 - B*rho vs plast sp2 for varying constant n2"
plotter_pops(data, 'sp2.b0', 'b1_del_thet', xlab, ylab, tit)

# NEED TO TRY SMOOTHING!!!