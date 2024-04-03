# -*- coding: utf-8 -*-
"""
Created on Tue May 16 13:21:21 2023

@author: laksh
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import numba as nb
import time
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings
import matplotlib as mpl

# Increase the path chunksize limit
mpl.rcParams['agg.path.chunksize'] = 1000  # You can adjust the value as needed

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)

# Deleted prim_sim, a primilinary simulation function no longer in use
# Deleted plot_bas, plot_rep and plot_tau as no longer used

# Forms the next two steps of autocorrelated environment values for varying tau values
# used in run_main_tau -> {further used in single_run, Exploration_VariedTau}
@nb.jit(nopython=True, nogil=True)
def env_tau(sel, rho, tau1, tau2, sig_eps):
    if tau1 > tau2:
        b = rho ** ((1 - tau1) / tau1) * sel + np.sqrt(1 - rho ** (2 * (1 - tau1) / tau1)) * np.random.normal(0, sig_eps)
        c = rho ** ((tau1 - tau2) / tau1) * b + np.sqrt(1 - rho ** (2 * (tau1 - tau2) / tau1)) * np.random.normal(0, sig_eps)
        d = rho ** ( tau2 / tau1) * c + np.sqrt(1 - rho ** (2 * tau2 / tau1)) * np.random.normal(0, sig_eps)
        return b, c, d 
    elif tau1 <= tau2:
        b = rho ** ((1 - tau2) / tau1) * sel + np.sqrt(1 - rho ** (2 * (1 - tau2) / tau1)) * np.random.normal(0, sig_eps)
        c = rho ** ((tau2 - tau1) / tau1) * b + np.sqrt(1 - rho ** (2 * (tau2 - tau1) / tau1)) * np.random.normal(0, sig_eps)
        d = rho * c + np.sqrt(1 - rho ** 2) * np.random.normal(0, sig_eps)
        return c, b, d

# Gives development time for a particular tau given previous selection time
# Used in exploration of things where taus is same for both species!
@nb.jit(nopython=True, nogil=True)
def dev_env(sel,rho,tau,sig_eps):
    return(rho ** ((1 - tau) / tau) * sel
    + np.sqrt(1 - rho ** (2 * (1 - tau) / tau)) * np.random.normal(0, sig_eps))

# Gives selection time for a particular tau given previous development time
# Used in exploration of things where taus is same for both species!
@nb.jit(nopython=True, nogil=True)
def sel_env(dev,rho,tau,sig_eps):
    return(rho * dev + np.sqrt(1 - rho ** 2) * np.random.normal(0, sig_eps))

# Calculates dist-from-optimal selection component of malthusian fitness
@nb.jit(nopython=True, nogil=True)
def mls_val(z1, theta1, sig_z, sig_s):
    return (-(sig_z**2 + (theta1 - z1)**2)/(2*sig_s**2))

# Calculates competition component of malthusian fitness
@nb.jit(nopython=True, nogil=True)
def mlc_val(z1, z2, n1, n2, r, kar, sig_u, sig_z):
    return (-r / kar) * np.sqrt(sig_u**2 / (sig_u**2 + sig_z**2)) * (n1 + n2 * np.exp(-(z1 - z2)**2 / (4 * (sig_u**2 + sig_z**2))))

# Calculates SG from dist-from-optimal selection component    
@nb.jit(nopython=True, nogil=True)
def sgs_val(z1, theta1, sig_s):
    return (theta1-z1)/(sig_s**2)

# Calculates SG from competition component 
@nb.jit(nopython=True, nogil=True)
def sgc_val(z1, z2, n2, r, kar, sig_u, sig_z):
    return (n2 / kar) * (np.exp(-(z1 - z2)**2 / (4 * (sig_u**2 + sig_z**2))) * r * (z1 - z2) * sig_u) / (2 * (sig_u**2 + sig_z**2)**(3/2))

# Calculates population level at next time point for species given
# malthusian fitness and grow[i] == 1
# species i's population remains constant if grow = [i] == 0
@nb.jit(nopython=True, nogil=True)
def pop_grow(n, grow, ml):
    return [np.exp(ml[0])*n[0] if grow[0] == 1 else n[0],
            np.exp(ml[1])*n[1] if grow[1] == 1 else n[1]]
   
# Plotting function of single runs given we are entertaining
# different tau's too for different species
def plot_tau_sep(a, b, n, sgs, sgc, epss, epsd1, epsd2, A, B, tot, tau, rho, subfold, fn):
    # plt.rcParams.update({'font.size': 11})
    
    # plt.suptitle(f"tau: {tau}")
    lw = 3
    figure, axis = plt.subplots(figsize=(10,8))
    # trait a
    axis.plot(range(0,tot), a[:,0], label = "species1", color='blue', linewidth=lw)
    axis.plot(range(0,tot), a[:,1], label = "species2", color='red', linewidth=lw)
    axis.legend(fontsize=19)
    axis.tick_params(axis='both', which='major', labelsize=19)
    axis.set_xlabel("Generations", fontsize=24)
    axis.set_ylabel("Population mean trait $a̅$", fontsize=24)
    plt.tight_layout()
    plt.show()
    # Save the figure
    dumsav = 'images/' + subfold + '/' + fn + "_a" + '.jpg'
    figure.savefig(dumsav, dpi=550)
    
    figure, axis = plt.subplots(figsize=(10,8))
    # trait b
    axis.plot(range(0,tot), b[:,0], label = "species1", color='blue', linewidth=lw)
    axis.plot(range(0,tot), b[:,1], label = "species2", color='red', linewidth=lw)
    # axis.axhline(B[0]*rho, color = 'blue', linestyle = 'dashed')
    # axis.axhline(B[1]*rho**(tau[1]/tau[0]), color = 'red', linestyle = 'dashed')
    # axis.legend(fontsize=19)
    axis.tick_params(axis='both', which='major', labelsize=19)
    axis.set_xlabel("Generations", fontsize=24)
    axis.set_ylabel("Mean Plasticity $b̅$", fontsize=24)
    plt.tight_layout()
    plt.show()
    # Save the figure
    dumsav = 'images/' + subfold + '/' + fn + "_b" + '.jpg'
    figure.savefig(dumsav, dpi=550)
    
    figure, axis = plt.subplots(figsize=(10,8))
    # Population size
    axis.plot(range(0,tot), n[:,0], label = "species1", color='blue', linewidth=lw)
    axis.plot(range(0,tot), n[:,1], label = "species2", color='red', linewidth=lw)
    axis.legend(fontsize=19)
    axis.tick_params(axis='both', which='major', labelsize=19)
    axis.set_xlabel("Generations", fontsize=24)
    axis.set_ylabel("Population size", fontsize=24)
    plt.tight_layout()
    plt.show()
    # Save the figure
    dumsav = 'images/' + subfold + '/' + fn + "_size" + '.jpg'
    figure.savefig(dumsav, dpi=550)
    
    figure, axis = plt.subplots(figsize=(10,8))
    # trait z
    axis.plot(range(0,tot), a[:,0] + b[:,0]*epsd1, label = "species1", color='blue', linewidth=lw)
    axis.plot(range(0,tot), a[:,1] + b[:,1]*epsd2, label = "species2", color='red', linewidth=lw)
    # axis.plot(range(0,tot), a[:,0], label = "a sp1", color='purple', linewidth=lw)
    # axis.plot(range(0,tot), a[:,1], label = "a sp2", color='black', linewidth=lw)
    axis.legend(fontsize=19)
    # axis.set_ylim(7,13)
    axis.tick_params(axis='both', which='major', labelsize=19)
    axis.set_xlabel("Generations", fontsize=24)
    axis.set_ylabel("Population mean phenotype $z̅$", fontsize=24)
    plt.tight_layout()
    plt.show()
    # Save the figure
    dumsav = 'images/' + subfold + '/' + fn + "_z" + '.jpg'
    figure.savefig(dumsav, dpi=550)
    
    figure, axis = plt.subplots(figsize=(10,8))
    # deltz
    axis.plot(range(0,tot), (a[:,0] + b[:,0]*epsd1) - a[:,1] + b[:,1]*epsd2, color='green', linewidth=lw)
    axis.legend(fontsize=19)
    axis.tick_params(axis='both', which='major', labelsize=19)
    axis.set_xlabel("Generations", fontsize=24)
    axis.set_ylabel("CD : $z̅_1$ - $z̅_2$", fontsize=24)
    plt.tight_layout()
    axis.legend()
    plt.show()
    # Save the figure
    dumsav = 'images/' + subfold + '/' + fn + "_delz" + '.jpg'
    figure.savefig(dumsav, dpi=550)

# Plotting function of single runs given we are entertaining
# different tau's too for different species
def plot_tau_sep_old(a, b, n, sgs, sgc, epss, epsd1, epsd2, A, B, tot, tau, rho):
    plt.rcParams.update({'font.size': 11})
    
    # plt.suptitle(f"tau: {tau}")
    lw = 1.2
    figure, axis = plt.subplots(figsize=(4,4))
    # trait a
    axis.plot(range(0,tot), a[:,0], label = "sp1", color='blue', linewidth=lw)
    axis.plot(range(0,tot), a[:,1], label = "sp2", color='red', linewidth=lw)
    # axis[0].set_ylim(np.min(a)*(1-0.1*np.sign(np.max(a))) , np.max(a)*(1+0.1*np.sign(np.min(a))))
    axis.legend(fontsize=12)
    # axis[0].set_title("trait a", fontsize=13)
    axis.set_xlabel("generations", fontsize=12.5)
    axis.set_ylabel("mean trait a", fontsize=12.5)
    # axis.set_ylim(7,13)
    # axis[0].set_title("trait a")
    plt.tight_layout()
    plt.show()
    
    figure, axis = plt.subplots(figsize=(4,4))
    # trait b
    axis.plot(range(0,tot), b[:,0], label = "sp1", color='blue', linewidth=lw)
    axis.plot(range(0,tot), b[:,1], label = "sp2", color='red', linewidth=lw)
    axis.axhline(B[0]*rho, color = 'blue', linestyle = 'dashed')
    axis.axhline(B[1]*rho**(tau[1]/tau[0]), color = 'red', linestyle = 'dashed')
    axis.legend(fontsize=12)
    # axis[0].set_title("trait a", fontsize=13)
    axis.set_xlabel("generations", fontsize=12.5)
    axis.set_ylabel("mean trait b", fontsize=12.5)
    plt.tight_layout()
    axis.legend()
    plt.show()
    
    figure, axis = plt.subplots(figsize=(4,4))
    # trait b
    axis.plot(range(0,tot), n[:,0], label = "sp1", color='blue', linewidth=lw)
    axis.plot(range(0,tot), n[:,1], label = "sp2", color='red', linewidth=lw)
    axis.legend(fontsize=12)
    # axis[0].set_title("trait a", fontsize=13)
    axis.set_xlabel("generations", fontsize=12.5)
    axis.set_ylabel("population size", fontsize=12.5)
    plt.tight_layout()
    axis.legend()
    plt.show()
    
    figure, axis = plt.subplots(figsize=(4,4))
    # trait z
    axis.plot(range(0,tot), a[:,0] + b[:,0]*epsd1, label = "z sp1", color='blue', linewidth=lw)
    axis.plot(range(0,tot), a[:,1] + b[:,1]*epsd2, label = "z sp2", color='red', linewidth=lw)
    axis.plot(range(0,tot), a[:,0], label = "a sp1", color='purple', linewidth=lw)
    axis.plot(range(0,tot), a[:,1], label = "a sp2", color='black', linewidth=lw)
    axis.legend(fontsize=12)
    # axis.set_ylim(7,13)
    # axis[0].set_title("trait a", fontsize=13)
    axis.set_xlabel("generations", fontsize=12.5)
    axis.set_ylabel("mean trait", fontsize=12.5)
    plt.tight_layout()
    axis.legend(fontsize=12)
    plt.show()
    
    figure, axis = plt.subplots(figsize=(4,4))
    # delta
    axis.plot(range(0,tot), (a[:,0] + b[:,0]*epsd1) - a[:,1] + b[:,1]*epsd2, color='green', linewidth=0.5)
    axis.legend(fontsize=12)
    # axis.set_title("z1 - z2")
    axis.legend(fontsize=12)
    # axis[0].set_title("trait a", fontsize=13)
    axis.set_xlabel("generations", fontsize=12.5)
    axis.set_ylabel("delta z", fontsize=12.5)
    plt.tight_layout()
    axis.legend()
    plt.show()

# Plots the three dynamic plots for each run in an exploration!!
def plot_tit(a, b, n, epsd, tot, a0, b0, n0, kar, A, B, rho):
    
    plt.rcParams.update({'font.size': 8})
    figure, axis = plt.subplots(3, 1, figsize=(3,8))
    figure.suptitle(f"a0: {a0}, b0: {b0}, n0: {n0}, kar: {kar}")
    
    # trait a
    axis[0].plot(range(0,tot), a[:,0], label = "a sp1", color='blue', linewidth=0.5)
    axis[0].plot(range(0,tot), a[:,1], label = "a sp2", color='red', linewidth=0.5)
    axis[0].axhline(y=np.unique(A), color='black')
    # axis[0].set_ylim(np.min(a)*(1-0.1*np.sign(np.max(a))) , np.max(a)*(1+0.1*np.sign(np.min(a))))
    axis[0].legend()
    axis[0].set_title("trait a")
      
    # trait b
    axis[1].plot(range(0,tot), b[:,0], label = "b sp1", color='blue', linewidth=0.5)
    axis[1].plot(range(0,tot), b[:,1], label = "b sp2", color='red', linewidth=0.5)
    axis[1].axhline(y=np.unique(B)*rho, color='black')
    axis[1].legend()
    axis[1].set_title("trait b")
      
    # population n
    axis[2].plot(range(0,tot), n[:,0], label = "n sp1", color='blue', linewidth=0.5)
    axis[2].plot(range(0,tot), n[:,1], label = "n sp2", color='red', linewidth=0.5)
    # axis[1].set_ylim(np.min(n)*(1-0.1*np.sign(np.max(n))) , np.max(n)*(1+0.1*np.sign(np.min(n))))
    axis[2].legend()
    axis[2].set_title("population n")
    
    # Delta z
    #axis[3].plot(range(0,tot), (a[:,0] + b[:,0]*epsd) - a[:,1] + b[:,1]*epsd, color='green', linewidth=0.5)
    #axis[1, 1].plot(range(0,tot), A[0] + B[0]*epss, label = "Th sp2", color='blue', linestyle='dashed')
    #axis[1, 1].plot(range(0,tot), A[1] + B[1]*epss, label = "Th sp2", color='red', linestyle='dashed')
    #axis[3].legend()
    #axis[3].set_title("z1 - z2")
    
    plt.tight_layout()
    # Combine all the operations and display
    plt.show()
    
# Simulation function for two different possible taus
# Used in {single_run.py AND exploration_variedTau}
def run_main_tau(a0, b0, n0, plast, grow, A, B, Gaa, Gbb, kar, rho, tau, r, sig_s, sig_u, sig_e, sig_eps, tot, seed):
    
    tau1 = tau[0]
    tau2 = tau[1]
    
    np.random.seed(int(time.time())) if seed < 0 else np.random.seed(seed)
    
    for i in range(0,2):
        if plast[i] <= 0: Gbb[i] = 0
        if plast[i] < 0: b0[i] = 0
        if plast[i] == -2: B[i] = 0

    a = [a0]
    b = [b0]
    n = [n0]
    mls = []
    mlc = []
    sgs = []
    sgc = []
    z = []
    theta = []
    dinit = np.random.normal(0, sig_eps)
    eps1 = [dinit]
    eps2 = [dinit]
    epss = []
    epsd1 = []
    epsd2 = []


    start = time.time()
    for i in range(tot):
        
        d1, d2, sng = env_tau(eps1[-1], rho, tau1, tau2, sig_eps)
        
        
        eps1.append( d1 )
        eps2.append( d2 )
        
        # eps1.append( dev_env( eps1[-1], rho, tau1, sig_eps))
        # eps2.append( dev_env( eps2[-1], rho, tau2, sig_eps))
        
        # sdum = sel_env( eps1[-1], rho, tau1, sig_eps)
        eps1.append(sng)
        eps2.append(sng)
        
        epsd1.append(eps1[-2])
        epsd2.append(eps2[-2])
        
        epss.append(sng)
        
        d_dum = np.array([eps1[-2], eps2[-2]])
        
        theta.append(list(A + B * epss[-1]))
        z.append(a[-1] + b[-1] * d_dum)
        sig_z = np.sqrt(Gaa + Gbb * d_dum ** 2 + sig_e ** 2)
        
        mls.append(np.array([mls_val( z[-1][0], theta[-1][0], sig_z[0], sig_s),
                            mls_val( z[-1][1], theta[-1][1], sig_z[1], sig_s)]))
        
        mlc.append(np.array([mlc_val(z[-1][0], z[-1][1], n[-1][0], n[-1][1], r, kar[0], sig_u, sig_z[0]),
                             mlc_val(z[-1][1], z[-1][0], n[-1][1], n[-1][0], r, kar[1], sig_u, sig_z[1])]))
        
        #ml.append(r + mls[-1] + mlc[-1])
        
        sgs.append(np.array([sgs_val( z[-1][0], theta[-1][0], sig_s),
                             sgs_val( z[-1][1], theta[-1][1], sig_s)]))
        
        sgc.append(np.array([sgc_val(z[-1][0], z[-1][1], n[-1][1], r, kar[0], sig_u, sig_z[0]),
                             sgc_val(z[-1][1], z[-1][0], n[-1][0], r, kar[1], sig_u, sig_z[1])]))
        
        #sg.append(sgs[-1]+sgc[-1])
        
        a.append(a[-1] + (sgs[-1] + sgc[-1]) * Gaa)
        b.append(b[-1] + (sgs[-1] + sgc[-1]) * Gbb * d_dum)
        n.append(pop_grow(n[-1], grow, r + mls[-1] + mlc[-1]))
    
    t_run = time.time() - start
    print(f'For loop: {t_run} seconds')
    
    a = np.array(a[1:])
    b = np.array(b[1:])
    n = np.array(n[1:])
    mls = np.array(mls)
    mlc = np.array(mlc)
    sgs = np.array(sgs)
    sgc = np.array(sgc)
    epss = np.array(epss)
    epsd1 = np.array(epsd1)
    epsd2 = np.array(epsd2)
    eps1 = eps1[1:]
    eps2 = eps2[1:]
    
    fin = 2000
    z1 = a[:,0] + b[:,0]*epsd1
    z2 = a[:,1] + b[:,1]*epsd2
    th1 = A[0] + B[0]*epss
    th2 = A[1] + B[1]*epss
    
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
    
    # traj.f_add_result('sp.$', a_=a, b_=b, n_=n, af = af_, bf = bf_, nf = nf_,
    #                   astd = astd_, bstd = bstd_, nstd = nstd_, 
    #                   mls=mls, mlc=mlc, sgs=sgs, sgc=sgc,
    #                   epss=epss, epsd=epsd, eps=eps,
    #                   comment='Contains all the development of various arrays')
    return a, b, n, mls, mlc, sgs, sgc, epss, epsd1, epsd2, t_run, eps1, eps2

# Fundamental Niche finder of an evolved population
def niche_finder_fund(af_, bf_, nf_, epsran, A, B, Gaa, Gbb, r, kar, sig_eps, sig_e, sig_s, sig_u):
    mlf1 = []
    mlf2 = []
    for i, e in enumerate(epsran):
        z1 = af_[0] + bf_[0]*e
        z2 = af_[1] + bf_[1]*e
        thet1 = A[0] + B[0]*e
        thet2 = A[1] + B[1]*e
        sig_z = Gaa + Gbb*sig_eps**2 + sig_e**2
        dum1 = mls_val(z1, thet1, sig_z[0], sig_s)
        mlf1.append(r + dum1)
        
        dum1 = mls_val(z2, thet2, sig_z[1], sig_s)
        mlf2.append(r + dum1)
    return mlf1, mlf2

# Niche finder of an evolved population under competition!!
def niche_finder_comp(af_, bf_, nf_, epsran, A, B, Gaa, Gbb, r, kar, sig_eps, sig_e, sig_s, sig_u):
    mlf1 = []
    mlf2 = []
    for i, e in enumerate(epsran):
        z1 = af_[0] + bf_[0]*e
        z2 = af_[1] + bf_[1]*e
        thet1 = A[0] + B[0]*e
        thet2 = A[1] + B[1]*e
        n1 = nf_[0]
        n2 = nf_[1]
        sig_z = Gaa + Gbb*sig_eps**2 + sig_e**2
        dum1 = mls_val(z1, thet1, sig_z[0], sig_s)
        dum2 = mlc_val(z1, z2, n1, n2, r, kar[0], sig_u, sig_z[0])
        # print(dum2)
        mlf1.append(r + dum1 + dum2)
        
        dum1 = mls_val(z2, thet2, sig_z[1], sig_s)
        dum2 = mlc_val(z2, z1, n2, n1, r, kar[1], sig_u, sig_z[1])
        mlf2.append(r + dum1 + dum2)
    return mlf1, mlf2