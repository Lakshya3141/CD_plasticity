# -*- coding: utf-8 -*-
"""
Created on Thu May 18 14:52:16 2023
@author: laksh
Code to run exploration simulations corresponding to Figure 5 of report
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
from helper_exploration import *

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)
 
generations = 300000

def add_param(traj):
    """Adds all parameters to `traj`"""
    print('Adding Parameters')

    # Following lines adds common paramaters
    traj.f_add_parameter('com.sig_e', np.sqrt(0.5).item(),
                         comment='Common phenotypic variance of both species')
    traj.f_add_parameter('com.sig_s', np.sqrt(300.0).item(),
                         comment='Strength of stabilising selection')
    traj.f_add_parameter('com.sig_u', np.sqrt(10.0).item(),
                         comment='Utilisation curve variance')
    traj.f_add_parameter('com.sig_eps', np.sqrt(2).item(),
                        comment='Strength of environmental fluctuations')
    traj.f_add_parameter('com.rho', 0.5,
                        comment='Autocorrelation between developmental'
                            'environment and selection environment')
    traj.f_add_parameter('com.tau', 0.5,
                        comment='fraction of generation between'
                            'development and selection')
    traj.f_add_parameter('com.r', 0.1,
                        comment='Growth rate')
    traj.f_add_parameter('com.seed', 0,
                        comment='Value of seed for choosing random values')
    traj.f_add_parameter('com.tot', generations,
                        comment='Number of generations ot run the simulation')
    
    
    # Following lines add species parameters
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
    traj.f_add_parameter('sp.grow', np.array([1, 0]),
                         comment='growth parameter')
    # plasticity parameter: -2 -> no fluctuations, -1 -> no plasticity
    #                       0 -> constant plasticity, 1 -> evolving plasticity
    traj.f_add_parameter('sp.plast', np.array([1, 0]),
                         comment='plasticity parameter')
    
# main function to add paramaters, add exploration, run simulations and store results
def param_explore(traj):
    """ Here is where you change all the kinda exploration you wanna do!"""
    print('Exploring across sig_s and sig_u')
    
    explore_dict = {'sp.b0': [np.array([2.51, i]) for i in np.arange(0, 2.501, 0.025)],
                    'sp.n0': [np.array([20000, i]) for i in np.arange(5000.0, 60000.0, 5000.0)]}
    
    explore_dict = cartesian_product(explore_dict, ('sp.b0','sp.n0'))
                    
    
    traj.f_explore(explore_dict)
    print('added exploration')

# main function to add paramaters, add exploration, run simulations and store results
def main(fn, fld, traje):
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
    add_param(traj)
    
    # Let's explore
    param_explore(traj)

    # Run the experiment
    env.run(run_main1)

    # Finally disable logging and close all log-files
    env.disable_logging()
    

if __name__ == '__main__':    

    fld = "dummy"
    
    fn = f"CDisp_Sp2CpCpop.hdf5"
    traje = 'dummy'
            
    main(fn, fld, traje)
    post_proc(fn, fld, traje)
