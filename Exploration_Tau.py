# -*- coding: utf-8 -*-
"""
Created on Thu May 18 14:52:16 2023
@author: laksh
Code to run exploration simulations corresponding to Figure 4 of report
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

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)

generations = 300000
bins = 100

def add_parameters_tau(traj, taus):

    
    """Adds all parameters to `traj`"""
    print('Adding Parameters')

    # Following lines adds common paramaters
    traj.f_add_parameter('com.sig_e', np.sqrt(0.5).item(),
                         comment='Common phenotypic variance of both species')
    traj.f_add_parameter('com.sig_s', np.sqrt(300.0).item(),
                         comment='Strength of stabilising selection')
    traj.f_add_parameter('com.sig_u', np.sqrt(10.0).item(),
                         comment='Utilisation curve variance')
    traj.f_add_parameter('com.sig_eps', np.sqrt(10).item(),
                        comment='Strength of environmental fluctuations')
    traj.f_add_parameter('com.rho', 0.5,
                        comment='Autocorrelation between developmental'
                            'environment and selection environment')
    traj.f_add_parameter('com.r', 0.1,
                        comment='Growth rate')
    traj.f_add_parameter('com.seed', 0,
                        comment='Value of seed for choosing random values')
    traj.f_add_parameter('com.tot', generations,
                        comment='Number of generations ot run the simulation')
    
    
    # Following lines add species parameters
    print(taus)
    # traj.f_add_parameter('sp.tau', np.array([taus[0], taus[1]]),
    #                     comment='fraction of generation between'
    #                         'development and selection for species 1 and 2')
    traj.f_add_parameter('com.tau1', np.array(taus[0]),
                        comment='fraction of generation between'
                            'development and selection for species 1')
    traj.f_add_parameter('com.tau2', np.array(taus[1]),
                        comment='fraction of generation between'
                            'development and selection for species 2')
    traj.f_add_parameter('sp.A', np.array([5.0, 5.0]),
                         comment='Optimal genetic trait value')
    traj.f_add_parameter('sp.B', np.array([3.0, 3.0]),
                         comment='Optimal plasticity')
    traj.f_add_parameter('sp.a0', np.array([5.3, 4.7]),
                         comment='Initial genetic trait value')
    traj.f_add_parameter('sp.b0', np.array([2.5, 2.51]),
                         comment='Initial plasticity value')
    traj.f_add_parameter('sp.kar', np.array([60000.0, 60000.0]),
                         comment='Carrrying capacities')
    traj.f_add_parameter('sp.n0', traj.kar/2,
                         comment='Inital populations, default half of carrying')
    traj.f_add_parameter('sp.Gaa', np.array([0.5, 0.5]),
                         comment='variance of trait a')
    traj.f_add_parameter('sp.Gbb', np.array([0.045, 0.045]),
                         comment='variance of trait b')
    # growth parameter: 0 -> static population, 1 -> growing population
    traj.f_add_parameter('sp.grow', np.array([1, 1]),
                         comment='growth parameter')
    # plasticity parameter: -2 -> no fluctuations, -1 -> no plasticity
    #                       0 -> constant plasticity, 1 -> evolving plasticity
    traj.f_add_parameter('sp.plast', np.array([1, 1]),
                         comment='plasticity parameter')
def param_exp_tau(traj):
    """ Here is where you change all the kinda exploration you wanna do!"""
    print('Exploring across sig_s and sig_u')
    
    explore_dict = {'sig_u': [np.sqrt(i).item() for i in np.arange(1.0, 81.0, 80.0/bins)],
                    'sig_s': [np.sqrt(i).item() for i in np.arange(1.0, 1225.0, 1224.0/bins)]}
    
    explore_dict = cartesian_product(explore_dict, ('sig_s','sig_u'))
                    
    
    traj.f_explore(explore_dict)
    print('added exploration')

def run_main_tau(traj):
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
    tau1 = traj.tau1
    tau2 = traj.tau2
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
    
    fin = 10000
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

def post_proc_tau(fn, fld, traje):
    
    traj = Trajectory(traje, add_time=False)
    traj.f_load(filename=os.path.join("hdf5",fld, fn), load_parameters=2,
                  load_results=2, load_derived_parameters=0, force=True)
    traj.v_auto_load = True
    res_list = traj.f_get_run_names()
    print('Starting post proc')
    col = ['sig_e','sig_s','sig_u','sig_eps','rho','tau1','tau2','r','seed',
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
        summary.iloc[runid]['tau1'] = traj.tau1
        summary.iloc[runid]['tau2'] = traj.tau2
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

def main(fn, fld, traje, taus):
    filename = os.path.join('hdf5', fld, fn)
    env = Environment(trajectory=traje,
                      comment='Setting up the pypet pipeline for our '
                            'temporal model of character displacement. ',
                      add_time=False, # We don't want to add the current time to the name,
                      log_stdout=True,
                      log_config='DEFAULT',
                      multiproc=True,
                      ncores=24,
                      wrap_mode='QUEUE',
                      filename=filename,
                      overwrite_file=True)
    traj = env.trajectory
    
    add_parameters_tau(traj, taus)
    
    # Let's explore
    param_exp_tau(traj)

    # Run the experiment
    env.run(run_main_tau)

    # Finally disable logging and close all log-files
    env.disable_logging()
    

if __name__ == '__main__':    
    
    tau_list = [[0.1, 0.5], [0.5, 0.9], [0.1, 0.9]]
    fn_ = ['LowDiff.hdf5','MidDiff.hdf5', 'HighDiff.hdf5']
    fld = "tau_exp_large"    
    
    for i, n in enumerate(tau_list):
        traje = 'dummy'
        print(f'starting {i+1} OUT OF 3')
        fn = fn_[i]
        main(fn, fld, traje, n)
        post_proc_tau(fn, fld, traje)
        print('finished')
