# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 14:03:54 2023

@author: laksh
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


bins = 2

def run_main0(traj):
    a0 = traj.a0
    b0 = traj.b0
    n0 = traj.n0
    plast = traj.plast
    grow = traj.grow
    A = traj.A
    B = traj.B
    Gaa = traj.Gaa
    Gbb = traj.Gbb
    kar = traj.kar
    rho = traj.rho
    tau = traj.tau
    r = traj.r
    sig_s = traj.sig_s
    sig_u = traj.sig_u
    sig_e = traj.sig_e
    sig_eps = traj.sig_eps
    tot = traj.tot
    seed = traj.seed
    
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
    eps = [np.random.normal(0, sig_eps)]
    epss = []
    epsd = []


    start = time.time()
    for i in range(tot):
        eps.append( dev_env( eps[-1], rho, tau, sig_eps))
        eps.append( sel_env( eps[-1], rho, tau, sig_eps))
        epsd.append(eps[-2])
        epss.append(eps[-1])
        theta.append(list(A + B * eps[-1]))
        z.append(a[-1] + b[-1] * eps[-2])
        sig_z = np.sqrt(Gaa + Gbb * eps[-2] ** 2 + sig_e ** 2)
        
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
        b.append(b[-1] + (sgs[-1] + sgc[-1]) * Gbb * epsd[-1])
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
    epsd = np.array(epsd)
    eps = eps[1:]
    
    traj.f_add_result('sp.$', a_=a, b_=b, n_=n,
                      epss=epss, epsd=epsd,
                      comment='Contains all the development of various arrays')
    
    # traj.f_add_result('sp.$', a_=a, b_=b, n_=n,
    #                   mls=mls, mlc=mlc, sgs=sgs, sgc=sgc,
    #                   epss=epss, epsd=epsd, eps=eps,
    #                   comment='Contains all the development of various arrays')
    #return a, b, n, mls, mlc, sgs, sgc, eps, epss, epsd, t_run
    return t_run

def run_main1(traj):
    a0 = traj.a0
    b0 = traj.b0
    n0 = traj.n0
    plast = traj.plast
    grow = traj.grow
    A = traj.A
    B = traj.B
    Gaa = traj.Gaa
    Gbb = traj.Gbb
    kar = traj.kar
    rho = traj.rho
    tau = traj.tau
    r = traj.r
    sig_s = traj.sig_s
    sig_u = traj.sig_u
    sig_e = traj.sig_e
    sig_eps = traj.sig_eps
    tot = traj.tot
    seed = traj.seed
    
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
    eps = [np.random.normal(0, sig_eps)]
    epss = []
    epsd = []


    start = time.time()
    for i in range(tot):
        eps.append( dev_env( eps[-1], rho, tau, sig_eps))
        eps.append( sel_env( eps[-1], rho, tau, sig_eps))
        epsd.append(eps[-2])
        epss.append(eps[-1])
        theta.append(list(A + B * eps[-1]))
        z.append(a[-1] + b[-1] * eps[-2])
        sig_z = np.sqrt(Gaa + Gbb * eps[-2] ** 2 + sig_e ** 2)
        
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
        b.append(b[-1] + (sgs[-1] + sgc[-1]) * Gbb * epsd[-1])
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
    epsd = np.array(epsd)
    eps = eps[1:]
    
    fin = 2000
    z1 = a[:,0] + b[:,0]*epsd
    z2 = a[:,1] + b[:,1]*epsd
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
    
    traj.f_add_result('sp.$', af = af_, bf = bf_, nf = nf_, zf = zf_,
                      astd = astd_, bstd = bstd_, nstd = nstd_, zstd = zstd_,
                      adf = adf_, bdf = bdf_, ndf = ndf_, zdf = zdf_,
                      adstd = adstd_, bdstd = bdstd_, ndstd = ndstd_, zdstd = zdstd_,
                      comment='Contains all final values and standard deviations')
    
    # traj.f_add_result('sp.$', a_=a, b_=b, n_=n, af = af_, bf = bf_, nf = nf_,
    #                   astd = astd_, bstd = bstd_, nstd = nstd_, 
    #                   mls=mls, mlc=mlc, sgs=sgs, sgc=sgc,
    #                   epss=epss, epsd=epsd, eps=eps,
    #                   comment='Contains all the development of various arrays')
    #return a, b, n, mls, mlc, sgs, sgc, eps, epss, epsd, t_run
    return t_run

def lin_compsel(traj):
    """ Here is where you change all the kinda exploration you wanna do!"""
    print('Exploring across sig_s and sig_u')
    
    explore_dict = {'sig_u': [i.item() for i in np.arange(1.0, 9.0, 8.0/100)],
                    'sig_s': [i.item() for i in np.arange(1.0, 35.0, 34.0/100)]}
    
    explore_dict = cartesian_product(explore_dict, ('sig_s','sig_u'))
                    
    
    traj.f_explore(explore_dict)
    print('added exploration')

def log_compsel(traj):
    """ Here is where you change all the kinda exploration you wanna do!"""
    print('Exploring across sig_s and sig_u')
    
    explore_dict = {'sig_u': [np.sqrt(i).item() for i in np.arange(1.0, 81.0, 80.0/bins)],
                    'sig_s': [np.sqrt(i).item() for i in np.arange(1.0, 1225.0, 1224.0/bins)]}
    
    explore_dict = cartesian_product(explore_dict, ('sig_s','sig_u'))
                    
    
    traj.f_explore(explore_dict)
    print('added exploration')
    
def post_proc(fn, fld, traje):
    
    traj = Trajectory(traje, add_time=False)
    traj.f_load(filename=os.path.join("hdf5",fld, fn), load_parameters=2,
                  load_results=2, load_derived_parameters=0, force=True)
    traj.v_auto_load = True
    res_list = traj.f_get_run_names()
    print('Starting post proc')
    col = ['sig_e','sig_s','sig_u','sig_eps','rho','tau','r','seed',
           'sp1.A','sp2.A','sp1.B','sp2.B','sp1.a0','sp2.a0','sp1.b0','sp2.b0',
           'sp1.n0','sp2.n0','sp1.kar','sp2.kar','sp1.Gaa','sp2.Gaa',
           'sp1.Gbb','sp2.Gbb','sp1.grow','sp2.grow','sp1.plast','sp2.plast','runid',
           'sp1.a', 'sp2.a', 'sp1.b', 'sp2.b', 'sp1.n', 'sp2.n', 'sp1.z', 'sp2.z',
           'sp1.astd', 'sp2.astd', 'sp1.bstd', 'sp2.bstd', 'sp1.nstd', 'sp2.nstd', 
           'sp1.zstd', 'sp2.zstd', 'dela', 'delb',
           'delz', 'delastd', 'delzstd', 'delbstd']
           #'divz','sp1.a','sp2.a','sp1.b','sp2.b','sp1.n','sp2.n']

    summary = pd.DataFrame(columns=col, index = res_list)
    # print(summary)
    for runid,run_name in enumerate(res_list):
        traj.f_set_crun(run_name)
        summary.iloc[runid]['sig_e'] = traj.sig_e
        summary.iloc[runid]['sig_eps'] = traj.sig_eps
        summary.iloc[runid]['sig_s'] = traj.sig_s
        summary.iloc[runid]['sig_u'] = traj.sig_u
        summary.iloc[runid]['rho'] = traj.rho
        summary.iloc[runid]['tau'] = traj.tau
        summary.iloc[runid]['r'] = traj.r
        summary.iloc[runid]['seed'] = traj.seed
        
        summary.iloc[runid]['sp1.A'] = traj.A[0]
        summary.iloc[runid]['sp2.A'] = traj.A[1]
        summary.iloc[runid]['sp1.B'] = traj.B[0]
        summary.iloc[runid]['sp2.B'] = traj.B[1]
        summary.iloc[runid]['sp1.a0'] = traj.a0[0]
        summary.iloc[runid]['sp2.a0'] = traj.a0[1]
        summary.iloc[runid]['sp1.b0'] = traj.b0[0]
        summary.iloc[runid]['sp2.b0'] = traj.b0[1]
        summary.iloc[runid]['sp1.n0'] = traj.n0[0]
        summary.iloc[runid]['sp2.n0'] = traj.n0[1]
        summary.iloc[runid]['sp1.kar'] = traj.kar[0]
        summary.iloc[runid]['sp2.kar'] = traj.kar[1]
        summary.iloc[runid]['sp1.Gaa'] = traj.Gaa[0]
        summary.iloc[runid]['sp2.Gaa'] = traj.Gaa[1]
        summary.iloc[runid]['sp1.Gbb'] = traj.Gbb[0]
        summary.iloc[runid]['sp2.Gbb'] = traj.Gbb[1]
        summary.iloc[runid]['sp1.grow'] = traj.grow[0]
        summary.iloc[runid]['sp2.grow'] = traj.grow[1]
        summary.iloc[runid]['sp1.plast'] = traj.plast[0]
        summary.iloc[runid]['sp2.plast'] = traj.plast[1]
        summary.iloc[runid]['runid'] = runid
        
        summary.iloc[runid]['sp1.a'] = traj.results.sp.crun.af[0]
        summary.iloc[runid]['sp2.a'] = traj.results.sp.crun.af[1]
        summary.iloc[runid]['sp1.b'] = traj.results.sp.crun.bf[0]
        summary.iloc[runid]['sp2.b'] = traj.results.sp.crun.bf[1]
        summary.iloc[runid]['sp1.n'] = traj.results.sp.crun.nf[0]
        summary.iloc[runid]['sp2.n'] = traj.results.sp.crun.nf[1]
        summary.iloc[runid]['sp1.z'] = traj.results.sp.crun.zf[0]
        summary.iloc[runid]['sp2.z'] = traj.results.sp.crun.zf[1]
        
        summary.iloc[runid]['sp1.astd'] = traj.results.sp.crun.astd[0]
        summary.iloc[runid]['sp2.astd'] = traj.results.sp.crun.astd[1]
        summary.iloc[runid]['sp1.bstd'] = traj.results.sp.crun.bstd[0]
        summary.iloc[runid]['sp2.bstd'] = traj.results.sp.crun.bstd[1]
        summary.iloc[runid]['sp1.nstd'] = traj.results.sp.crun.nstd[0]
        summary.iloc[runid]['sp2.nstd'] = traj.results.sp.crun.nstd[1]
        summary.iloc[runid]['sp1.zstd'] = traj.results.sp.crun.zstd[0]
        summary.iloc[runid]['sp2.zstd'] = traj.results.sp.crun.zstd[1]
        
        summary.iloc[runid]['dela'] = traj.results.sp.crun.adf
        summary.iloc[runid]['delb'] = traj.results.sp.crun.bdf
        summary.iloc[runid]['delz'] = traj.results.sp.crun.zdf
        summary.iloc[runid]['delastd'] = traj.results.sp.crun.adstd
        summary.iloc[runid]['delbstd'] = traj.results.sp.crun.bdstd
        summary.iloc[runid]['delzstd'] = traj.results.sp.crun.zdstd
        
        traj.f_restore_default()
    traj.f_add_result('sm',  data=summary,
                      comment='Summary table with all parameters and results')
    summary.to_csv(os.path.join("hdf5",fld, fn[0:-5]+'.csv'))
    print(os.path.join("hdf5",fld, fn[0:-5]+'.csv'))
    traj.f_store()
    
def post_plot(traj, run_list):
    print('Starting plotting')
    
    for run_name in run_list:
        traj.f_set_crun(run_name)
        a = traj.results.sp.crun.a_
        b = traj.results.sp.crun.b_
        n = traj.results.sp.crun.n_
        epsd = traj.results.sp.crun.epsd
        a0 = traj.a0
        b0 = traj.b0
        n0 = traj.n0
        tot = traj.tot
        kar = traj.kar
        A = traj.A
        B = traj.B
        rho = traj.rho
        print(run_name)
        plot_tit(a, b, n, epsd, tot, a0, b0, n0, kar, A, B, rho)
        traj.f_restore_default()
    
    print('Done plot')
