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
import os
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import Rectangle
import math


warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)

#%% Base definitions
def organize_simulation_output(a, b, n, mls, mlc, sgs, sgc, epss, epsd1, epsd2, t_run, eps1, eps2):
    output = {
        "a": a,
        "b": b,
        "n": n,
        "mean_phenotype_species1": mls,
        "mean_phenotype_species2": mlc,
        "genetic_variance_species1": sgs,
        "genetic_variance_species2": sgc,
        "environmental_variance": epss,
        "environmental_deviation_species1": epsd1,
        "environmental_deviation_species2": epsd2,
        "generation_number": t_run,
        "environmental_variance_species1": eps1,
        "environmental_variance_species2": eps2
    }

    return output

def no_fluc_CD(sigma_u, sigma_z, sigma_s, r, N, K):
    term1 = sigma_u**2 + sigma_z**2
    term2 = (sigma_u * sigma_s**2) / (sigma_u**2 + sigma_z**2)**(3/2)
    inner_sqrt = term1 * math.log((r * N / K) * term2)
    result = math.sqrt(inner_sqrt)
    return result

def NoFluc_noPop(sigma_u, sigma_z, sigma_s, r):
    numerator = 2 * sigma_s**2 * r - sigma_z**2
    denominator = 4 * (sigma_u**2 + sigma_z**2)
    inner_log = numerator / denominator
    if inner_log <= 0:
        return None  # Return None for imaginary results
    else:
        result = math.sqrt((sigma_u**2 + sigma_z**2) * math.log(inner_log))
        return result
    
def Fluc_noPop(sigma_u, sigma_z, sigma_s, r, Bi, sigma_eps):
    numerator = 2 * sigma_s**2 * r - sigma_z**2 - (Bi**2)*(sigma_eps**2)
    denominator = 4 * (sigma_u**2 + sigma_z**2)
    inner_log = numerator / denominator
    if inner_log <= 0:
        return None  # Return None for imaginary results
    else:
        result = math.sqrt((sigma_u**2 + sigma_z**2) * math.log(inner_log))
        return result

def Plas_noPop(sigma_u, sigma_z, sigma_s, r, Bi, sigma_eps, rho):
    numerator = 2 * sigma_s**2 * r - sigma_z**2 - (Bi**2)*(sigma_eps**2)*(1-rho**2)
    denominator = 4 * (sigma_u**2 + sigma_z**2)
    inner_log = numerator / denominator
    if inner_log <= 0:
        return None  # Return None for imaginary results
    else:
        result = math.sqrt((sigma_u**2 + sigma_z**2) * math.log(inner_log))
        return result


A = np.array([5, 5]) # optimal value of trait
B = np.array([3, 3]) # optimal value of plasticity under constant environment
b0 = np.array([0.01, 0.0]) # initial value for plasticity
n0 = np.array([10000.0, 10000.0]) # initial populations for sp1 and 2 respectively

Gaa = np.array([0.5, 0.5]) # additive genetic variance for intercept a
Gbb = np.array([0.045, 0.045]) # additive genetic variance for plasticity b
sig_e = np.sqrt(0.5) # noise in phenotype
sig_s = np.sqrt(550) # width of stabilising selection curve
sig_eps = np.sqrt(0) # variance in environmental fluctuations
sig_u = np.sqrt(10) # width of resource utilization curve
rho = 0.5  # Correlation between development and selection for focal species
tau = [0.3, 0.3] # time beteen maturation and development for species 1 and 2 respectively

 # total number of generations to run for
r = 0.1 # intrinsic growth rate
kar = np.array([60000.0, 60000.0])  # Carrying capacity
# -2: No fluct, -1: no plast, 0: constant plast, 1: evolving plasticity
grow = np.array([1, 1])  # 0 for no growth and 1 for growth of species 2
seed = 0 # random seed to be used while generating fluctuating env.

lw = 3
avg_fin = 1000
subfold = "single_final"
# Create 'images' folder if it doesn't exist
if not os.path.exists(os.path.join('images', subfold)):
    os.makedirs(os.path.join('images', subfold))
    
#%% No fluctuations plasticity images
plast = np.array([-2, -2])
tot = 8000
CD_point = 7000
CDnp_point = 7500
a0 = np.array([5.01, 4.99]) # initial value for reaction norm elevation
# a0 = np.array([13, 7])
sig_eps = np.sqrt(4)
fn = "alpha"

a, b, n, mls, mlc, sgs, sgc, epss, epsd1, epsd2, t_run, eps1, eps2 = run_main_tau(a0, b0, n0, plast, grow, 
                                                                                    A, B, Gaa, Gbb, kar, rho, tau, r, 
                                                                                    sig_s, sig_u, sig_e, sig_eps, tot, seed)
alpha = organize_simulation_output(a, b, n, mls, mlc, sgs, sgc, epss, epsd1, epsd2, t_run, eps1, eps2)


n1 = np.mean(n[-avg_fin:,0])
n2 = np.mean(n[-avg_fin:,1])
sig_z1 = np.mean(np.sqrt(Gaa[0] + Gbb[0] * epss[-avg_fin:] ** 2 + sig_e ** 2))
sig_z2 = np.mean(np.sqrt(Gaa[1] + Gbb[1] * epss[-avg_fin:] ** 2 + sig_e ** 2))

CD1 = no_fluc_CD(sig_u, sig_z1, sig_s, r, n1, kar[0])
CD2 = no_fluc_CD(sig_u, sig_z2, sig_s, r, n2, kar[1])
CDnp1 = NoFluc_noPop(sig_u, sig_z1, sig_s, r)
CDnp2 = NoFluc_noPop(sig_u, sig_z2, sig_s, r)


figure, axis = plt.subplots(figsize=(10,8))
# trait a
# Draw horizontal dotted line at A[0] spanning the entire x-axis range
plt.hlines(y=A[0], xmin=0, xmax=tot/1000, linestyle='--', color='black', label='optimum', linewidth=lw)

axis.plot(np.arange(0, tot/1000, 0.001), alpha['a'][:,0], label="species1", color='blue', linewidth=lw)
axis.plot(np.arange(0, tot/1000, 0.001), alpha['a'][:,1], label="species2", color='red', linewidth=lw)

# Add arrow to CD1
axis.annotate('', xy=(CD_point/1000, A[0]), xytext=(CD_point/1000, A[0]+CD1), arrowprops=dict(arrowstyle='<-', linestyle = (0, (5, 4)), linewidth=2, mutation_scale=30, color = "blue"))
axis.annotate('', xy=(CDnp_point/1000, A[0]), xytext=(CDnp_point/1000, A[0]+CDnp1), arrowprops=dict(arrowstyle='<-', linestyle = (0, (1, 5)), linewidth=2, mutation_scale=30, color = "blue"))

# Add arrow to CD2
axis.annotate('', xy=(CD_point/1000, A[0]), xytext=(CD_point/1000, A[0]-CD2), arrowprops=dict(arrowstyle='<-', linestyle = (0, (5, 4)), linewidth=2, mutation_scale=30, color = "red"))
axis.annotate('', xy=(CDnp_point/1000, A[0]), xytext=(CDnp_point/1000, A[0]-CDnp2), arrowprops=dict(arrowstyle='<-', linestyle = (0, (1, 5)), linewidth=2, mutation_scale=30, color = "red"))

axis.legend(fontsize=26)
axis.tick_params(axis='both', which='major', labelsize=26)
axis.set_xlabel("Generations (x1000)", fontsize=36, labelpad=15)
axis.set_ylabel("Mean Phenotype $z̅$", fontsize=36, labelpad=15)
plt.tight_layout()
plt.show()
dumsav = 'images/' + subfold + '/' + fn + '.jpg'
figure.savefig(dumsav, dpi=550)


#%% Non evolvable plasticity images Low Sig_eps
plast = np.array([-1, -1])
tot = 20000
CD_point = 17500
CDnp_point = 18500
a0 = np.array([5.01, 4.99]) # initial value for reaction norm elevation
sig_eps = np.sqrt(4)
fn = "beta"

a, b, n, mls, mlc, sgs, sgc, epss, epsd1, epsd2, t_run, eps1, eps2 = run_main_tau(a0, b0, n0, plast, grow, 
                                                                                    A, B, Gaa, Gbb, kar, rho, tau, r, 
                                                                                    sig_s, sig_u, sig_e, sig_eps, tot, seed)
alpha = organize_simulation_output(a, b, n, mls, mlc, sgs, sgc, epss, epsd1, epsd2, t_run, eps1, eps2)

n1 = np.mean(n[-avg_fin:,0])
n2 = np.mean(n[-avg_fin:,1])
sig_z1 = np.mean(np.sqrt(Gaa[0] + Gbb[0] * epss[-avg_fin:] ** 2 + sig_e ** 2))
sig_z2 = np.mean(np.sqrt(Gaa[1] + Gbb[1] * epss[-avg_fin:] ** 2 + sig_e ** 2))

CD1 = no_fluc_CD(sig_u, sig_z1, sig_s, r, n1, kar[0])
CD2 = no_fluc_CD(sig_u, sig_z2, sig_s, r, n2, kar[1])
CDnp1 = Fluc_noPop(sig_u, sig_z1, sig_s, r, B[0], sig_eps)
CDnp2 = Fluc_noPop(sig_u, sig_z2, sig_s, r, B[1], sig_eps)

figure, axis = plt.subplots(figsize=(10,8))
# trait a
# Draw horizontal dotted line at A[0] spanning the entire x-axis range
plt.hlines(y=A[0], xmin=0, xmax=tot/1000, linestyle='--', color='black', label='Optima', linewidth=lw)

axis.plot(np.arange(0, tot/1000, 0.001), alpha['a'][:,0], label="species1", color='blue', linewidth=lw)
axis.plot(np.arange(0, tot/1000, 0.001), alpha['a'][:,1], label="species2", color='red', linewidth=lw)

# Add arrow to CD1
axis.annotate('', xy=(CD_point/1000, A[0]), xytext=(CD_point/1000, A[0]+CD1), arrowprops=dict(arrowstyle='<-', linestyle = (0, (5, 4)), linewidth=2, mutation_scale=30, color = "blue"))
axis.annotate('', xy=(CDnp_point/1000, A[0]), xytext=(CDnp_point/1000, A[0]+CDnp1), arrowprops=dict(arrowstyle='<-', linestyle = (0, (1, 5)), linewidth=2, mutation_scale=30, color = "blue"))

# Add arrow to CD2
axis.annotate('', xy=(CD_point/1000, A[0]), xytext=(CD_point/1000, A[0]-CD2), arrowprops=dict(arrowstyle='<-', linestyle = (0, (5, 4)), linewidth=2, mutation_scale=30, color = "red"))
axis.annotate('', xy=(CDnp_point/1000, A[0]), xytext=(CDnp_point/1000, A[0]-CDnp2), arrowprops=dict(arrowstyle='<-', linestyle = (0, (1, 5)), linewidth=2, mutation_scale=30, color = "red"))

# axis.legend(fontsize=26)
axis.tick_params(axis='both', which='major', labelsize=26)
axis.set_xlabel("Generations (x1000)", fontsize=36, labelpad=15)
axis.set_ylabel("Mean Phenotype $z̅$", fontsize=36, labelpad=15)
plt.tight_layout()
plt.show()
dumsav = 'images/' + subfold + '/' + fn + '.jpg'
figure.savefig(dumsav, dpi=550)

#%% Non evolvable plasticity images High Sig_eps
plast = np.array([-1, -1])
tot = 30000
CD_point = 17500
a0 = np.array([7.8, 2.2]) # initial value for reaction norm elevation
sig_eps = np.sqrt(8)
fn = "gamma"

a, b, n, mls, mlc, sgs, sgc, epss, epsd1, epsd2, t_run, eps1, eps2 = run_main_tau(a0, b0, n0, plast, grow, 
                                                                                    A, B, Gaa, Gbb, kar, rho, tau, r, 
                                                                                    sig_s, sig_u, sig_e, sig_eps, tot, seed)
alpha = organize_simulation_output(a, b, n, mls, mlc, sgs, sgc, epss, epsd1, epsd2, t_run, eps1, eps2)

n1 = np.mean(n[-avg_fin:,0])
n2 = np.mean(n[-avg_fin:,1])
sig_z1 = np.mean(np.sqrt(Gaa[0] + Gbb[0] * epss[-avg_fin:] ** 2 + sig_e ** 2))
sig_z2 = np.mean(np.sqrt(Gaa[1] + Gbb[1] * epss[-avg_fin:] ** 2 + sig_e ** 2))

figure, axis = plt.subplots(figsize=(10,8))

axis.plot(np.arange(0, tot/1000, 0.001), alpha['a'][:,0], label="species1", color='blue', linewidth=lw)
axis.plot(np.arange(0, tot/1000, 0.001), alpha['a'][:,1], label="species2", color='red', linewidth=lw)
plt.hlines(y=A[0], xmin=0, xmax=tot/1000, linestyle='--', color='black', label='Optima', linewidth=lw)


# axis.legend(fontsize=26)
axis.tick_params(axis='both', which='major', labelsize=26)
axis.set_xlabel("Generations (x1000)", fontsize=36, labelpad=15)
axis.set_ylabel("Mean Phenotype $z̅$", fontsize=36, labelpad=15)
plt.tight_layout()
plt.show()
dumsav = 'images/' + subfold + '/' + fn + '.jpg'
figure.savefig(dumsav, dpi=550)
#%% Evolvable plasticity images
plast = np.array([1, 1])
tot = 60000
CD_point = 52000
CDnp_point = 56000
a0 = np.array([5.01, 4.99]) # initial value for reaction norm elevation
sig_eps = np.sqrt(8)
fn = "delta"

a, b, n, mls, mlc, sgs, sgc, epss, epsd1, epsd2, t_run, eps1, eps2 = run_main_tau(a0, b0, n0, plast, grow, 
                                                                                    A, B, Gaa, Gbb, kar, rho, tau, r, 
                                                                                    sig_s, sig_u, sig_e, sig_eps, tot, seed)
alpha = organize_simulation_output(a, b, n, mls, mlc, sgs, sgc, epss, epsd1, epsd2, t_run, eps1, eps2)

n1 = np.mean(n[-avg_fin:,0])
n2 = np.mean(n[-avg_fin:,1])
sig_z1 = np.mean(np.sqrt(Gaa[0] + Gbb[0] * epss[-avg_fin:] ** 2 + sig_e ** 2))
sig_z2 = np.mean(np.sqrt(Gaa[1] + Gbb[1] * epss[-avg_fin:] ** 2 + sig_e ** 2))

CD1 = no_fluc_CD(sig_u, sig_z1, sig_s, r, n1, kar[0])
CD2 = no_fluc_CD(sig_u, sig_z2, sig_s, r, n2, kar[1])
CDnp1 = Plas_noPop(sig_u, sig_z1, sig_s, r, B[0], sig_eps, rho)
CDnp2 = Plas_noPop(sig_u, sig_z2, sig_s, r, B[1], sig_eps, rho)

# Create the main plot
fig, axis = plt.subplots(figsize=(10,8))

plt.hlines(y=A[0], xmin=0, xmax=tot/1000, linestyle='--', color='black', label='Optima', linewidth=lw)
# Plot the main data
axis.plot(np.arange(0,tot/1000,0.001), alpha['a'][:,0], color='blue', label = "$a̅$ species1", linewidth=lw)
axis.plot(np.arange(0,tot/1000,0.001), alpha['a'][:,1], color='red', label = "$a̅$ species2", linewidth=lw)

# Set labels and legends
# axis.legend(fontsize=22, ncol=2)
axis.tick_params(axis='both', which='major', labelsize=26)
axis.set_xlabel("Generations (x1000)", fontsize=36)
axis.set_ylabel("Phenotype in \ncommon garden ($a̅$)", fontsize=36)
axis.plot(np.arange(0, tot/1000, 0.001), alpha['a'][:,0], label="species1", color='blue', linewidth=lw)
axis.plot(np.arange(0, tot/1000, 0.001), alpha['a'][:,1], label="species2", color='red', linewidth=lw)
plt.hlines(y=A[0], xmin=0, xmax=tot/1000, linestyle='--', color='black', label='Optima', linewidth=lw)

# Add arrow to CD1
axis.annotate('', xy=(CD_point/1000, A[0]), xytext=(CD_point/1000, A[0]+CD1), arrowprops=dict(arrowstyle='<-', linestyle = (0, (5, 4)), linewidth=2, mutation_scale=30, color = "blue"))
axis.annotate('', xy=(CDnp_point/1000, A[0]), xytext=(CDnp_point/1000, A[0]+CDnp1), arrowprops=dict(arrowstyle='<-', linestyle = (0, (1, 5)), linewidth=2, mutation_scale=30, color = "blue"))

# Add arrow to CD2
axis.annotate('', xy=(CD_point/1000, A[0]), xytext=(CD_point/1000, A[0]-CD2), arrowprops=dict(arrowstyle='<-', linestyle = (0, (5, 4)), linewidth=2, mutation_scale=30, color = "red"))
axis.annotate('', xy=(CDnp_point/1000, A[0]), xytext=(CDnp_point/1000, A[0]-CDnp2), arrowprops=dict(arrowstyle='<-', linestyle = (0, (1, 5)), linewidth=2, mutation_scale=30, color = "red"))

# Create the inset plot
inset_axis = inset_axes(axis, width="23%", height="23%", loc='lower left', borderpad=6.5)
inset_axis.set_facecolor('white')  # Set background color to white
                
# Plot the inset data
inset_axis.plot(np.arange(0,tot/1000,0.001), alpha['a'][:,0] + alpha['b'][:,0]*alpha["environmental_deviation_species1"], color='blue', linewidth=2, alpha = 0.5) #, label = "$z̅$ species1", color='green'
inset_axis.plot(np.arange(0,tot/1000,0.001), alpha['a'][:,1] + alpha['b'][:,1]*alpha["environmental_deviation_species2"], color='red', linewidth=2, alpha = 0.5) #, label = "$z̅$ species2", color='orange'

# Set labels and legends for inset plot
inset_axis.tick_params(axis='both', which='major', labelsize=16)
inset_axis.set_xlabel("Generations (x1000)", fontsize=18)
inset_axis.set_ylabel("Phenotype $z̅$", fontsize=18)
# inset_axis.legend(fontsize=16, loc='upper left')

inset_rect = Rectangle((0, 0), 1, 1, fill=True, color='white', alpha=1, zorder=100)
inset_axis.add_patch(inset_rect)
# Save the figure

plt.tight_layout()
plt.show()
dumsav = 'images/' + subfold + '/' + fn + '.jpg'
# fig.savefig(dumsav, dpi=550)

## Plasticity image Fig 1E ## 
fig, axis = plt.subplots(figsize=(10,8))

# Set labels and legends

axis.tick_params(axis='both', which='major', labelsize=26)
axis.set_xlabel("Generations (x1000)", fontsize=36)
axis.set_ylabel("Mean plastcity ($b̅$)", fontsize=36)
axis.plot(np.arange(0, tot/1000, 0.001), alpha['b'][:,0], linewidth=lw, color='blue')
axis.plot(np.arange(0, tot/1000, 0.001), alpha['b'][:,1], linewidth=lw, color='red')
plt.hlines(y=B[0]*rho, xmin=0, xmax=tot/1000, linestyle = (0, (1, 3)), color='black', label='expected \nplasticity', linewidth=lw)
axis.legend(fontsize=26)
plt.tight_layout()
plt.show()
dumsav = 'images/' + subfold + '/' + 'delta_plasticity.jpg'
fig.savefig(dumsav, dpi=550)