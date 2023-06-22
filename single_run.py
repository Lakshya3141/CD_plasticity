# -*- coding: utf-8 -*-
"""
Created on Thu May 18 14:52:16 2023
@author: laksh
Use this code to run a single simulation and plot dynamics and niche
"""
import matplotlib.pyplot as plt
import numpy as np
from helper_funcs import *
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)

#%% Defining paramaters to be used
A = np.array([5, 5]) # optimal value of trait
B = np.array([3, 3]) # optimal value of plasticity under constant environment
a0 = np.array([5.2, 4.8]) # initial value for reaction norm elevation
b0 = np.array([2.5, 2.51]) # initial value for plasticity
n0 = np.array([10000.0, 10000.0]) # initial populations for sp1 and 2 respectively

Gaa = np.array([0.5, 0.5]) # additive genetic variance for intercept a
Gbb = np.array([0.045, 0.045]) # additive genetic variance for plasticity b
sig_e = np.sqrt(0.5) # noise in phenotype
sig_s = np.sqrt(1000) # width of stabilising selection curve
sig_eps = np.sqrt(2) # variance in environmental fluctuations
sig_u = np.sqrt(10) # width of resource utilization curve
rho = 0.5  # Correlation between development and selection for focal species
tau = [0.1, 0.9] # time beteen maturation and development for species 1 and 2 respectively

tot = 100000 # total number of generations to run for
r = 0.1 # intrinsic growth rate
kar = np.array([60000.0, 60000.0])  # Carrying capacity
# -2: No fluct, -1: no plast, 0: constant plast, 1: evolving plasticity
plast = np.array([1, 1])  
grow = np.array([1, 1])  # 0 for no growth and 1 for growth of species 2
seed = 0 # random seed to be used while generating fluctuating env.

#%% Running the main simulation
# the function run_main_tau in helper_funcs runs the simulation using the 
# parameters defined above and returns various arrays
a, b, n, mls, mlc, sgs, sgc, epss, epsd1, epsd2, t_run, eps1, eps2 = run_main_tau(a0, b0, n0, plast, grow, 
                                                                A, B, Gaa, Gbb, kar, rho, tau, r, 
                                                                sig_s, sig_u, sig_e, sig_eps, tot, seed)

#%% plotting of dynamics
# uses the plot_tau_sep in helper_funcs to plot abar, bbar, population, zbar and delz
plot_tau_sep(a, b, n, sgs, sgc, epss, epsd1, epsd2, A, B, tot, tau, rho)

#%% Finding evolved niche
fin = 10000 #number of last generations to average over
z1 = a[:,0] + b[:,0]*epsd1
z2 = a[:,1] + b[:,1]*epsd2
th1 = A[0] + B[0]*epss
th2 = A[1] + B[1]*epss

# Averages over last <fin> number of generations
af_ = np.mean(a[-fin:-1,:], axis = 0)
bf_ = np.mean(b[-fin:-1,:], axis = 0)
nf_ = np.mean(n[-fin:-1,:], axis = 0)
zf_ = np.array([np.mean(z1[-fin:-1]), np.mean(z2[-fin:-1])])
astd_ = np.std(a[-fin:-1,:], axis = 0)
bstd_ = np.std(b[-fin:-1,:], axis = 0)
nstd_ = np.std(n[-fin:-1,:], axis = 0)
zstd_ = np.array([np.std(z1[-fin:-1]), np.std(z2[-fin:-1])])

adf_= abs(np.mean(a[-fin:-1,0] - a[-fin:-1,1]))
adstd_ = np.std(a[-fin:-1,0] - a[-fin:-1,1])
bdf_= abs(np.mean(b[-fin:-1,0] - b[-fin:-1,1]))
bdstd_ = np.std(b[-fin:-1,0] - b[-fin:-1,1])
ndf_= abs(np.mean(n[-fin:-1,0] - n[-fin:-1,1]))
ndstd_ = np.std(n[-fin:-1,0] - n[-fin:-1,1])
zdf_= abs(np.mean(z1[-fin:-1] - z2[-fin:-1]))
zdstd_ = np.std(z1[-fin:-1] - z2[-fin:-1])

# range of environments to run through
stder = 6 # st deviations to plot on each side
epsran = np.arange(-stder*sig_eps, stder*sig_eps, 0.01)

# evolved niche in presence of competition
mlf1, mlf2 = niche_finder_fund(af_, bf_, nf_, epsran, A, B, Gaa, Gbb, r, kar, sig_eps, sig_e, sig_s, sig_u)
# evolved fundamental niche
mlfc1, mlfc2 = niche_finder_comp(af_, bf_, nf_, epsran, A, B, Gaa, Gbb, r, kar, sig_eps, sig_e, sig_s, sig_u)
# niche of initial species
mli1, mli2 = niche_finder_fund(A, B, n0, epsran, A, B, Gaa, Gbb, r, kar, sig_eps, sig_e, sig_s, sig_u)

#plotting niche
fig, ax = plt.subplots(figsize=(5,4))
plt.xlabel("Environment")
plt.ylabel("Malthusian fitness")
plt.plot(epsran, mlf1, label="evolved sp1", c='blue')
plt.plot(epsran, mlf2, label="evolved sp2", c='red')
plt.plot(epsran, mli1, label="initial sp", c='green')
# plt.plot(epsran, mlfc1, label="realized sp1", c='blue', linestyle='dashed')
# plt.plot(epsran, mlfc2, label="realized sp2", c='red', linestyle='dashed')
# plt.plot(epsran, mli1, label="init sp1")
plt.tight_layout()
plt.legend()