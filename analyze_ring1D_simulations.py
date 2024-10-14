#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 15:34:38 2024

@author: sarah
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.stats as st
import pandas as pd
#import glob
import os.path
import datetime
import ring_1D2a as simu
import decode
import scipy
from scipy.optimize import curve_fit

# import saved simulation outputs
# these files have no inhibition gain applied during effort

filenames = ['simulation_outputs/no_effort_sim_with_inhibition_without_attention.csv',
    'simulation_outputs/no_effort_sim_with_inhibition_with_attention.csv',
    'simulation_outputs/effort_sim_without_attention.csv',
    'simulation_outputs/effort_sim_with_attention.csv',]

# these files do
#filenames = ['simulation_outputs/no_effort_sim_with_inhibition_without_attention.csv',
#    'simulation_outputs/no_effort_sim_with_inhibition_with_attention.csv', 
#    'simulation_outputs/effort_sim_with_inhibition_without_attention.csv',
#    'simulation_outputs/effort_sim_with_inhibition_with_attention.csv',]

MSE = np.zeros(len(filenames))
amplitudes = np.zeros(len(filenames))
SDs = np.zeros(len(filenames))

# cycle over different files to pull main measures from each of them
for simii in np.arange(len(filenames)):
    
    end_bumps = np.loadtxt(filenames[simii],dtype=float,delimiter=',')
    
    nsims = end_bumps.shape[0]
    N = end_bumps.shape[1]
    
    # establish some basics about the parameters of the previous simulation
    inpPeak = N//2
    thetafunc = decode.gettheta
    theta = thetafunc(np.arange(0,N),N)    
    
    # DECODE POSITION AND CORRECTNESS from AVERAGED activity from LAST 50 seconds
    # of DELAY PERIOD #
     
    # decoding is decoding error
    decoding = np.apply_along_axis(lambda x: decode.decoding1D(x,inpPeak,ori=theta), 1, end_bumps)
    # decoding_pos is the decoded location
    decoding_pos = np.apply_along_axis(lambda x: decode.decoding1D(x,ori=theta), 1, end_bumps)
    
    # GET MEASURES FROM THESE DECODING RESULTS
     
    # calculate MSE across simulations
    MSE[simii] = np.mean(decoding**2)
    print('mean squared error: ' + str(MSE[simii]))
    
    # FIRST: BE ASSUMPTION-FREE ABOUT THE SHAPE OF THE DISTRIBUTION
    # get amplitude out of un-normalized activation bumps
    mean_amplitude = np.mean(np.max(end_bumps,axis=1))
    print('mean amplitude of bump: ' + str(mean_amplitude))
    
    # normalize activity bumps from each trial (remove variations in amplitude)
    denominator = np.outer(np.sum(end_bumps,axis=1),np.ones((1,N)))
    normalized_bumps = end_bumps/denominator
    
    # obtain circular means of bump here
    individual_means = decoding_pos
    xs = np.arange(-180,180)
    
    # obtain empirical standard deviation (variance) here
    individual_SDs = [np.sqrt(np.sum(((xs-individual_means[tii])**2)*normalized_bumps[tii,:])) for tii in np.arange(nsims)]
    print('mean EMPIRICAL standard deviation of bump: ' + str(np.mean(individual_SDs)))
    
    # SECOND: ASSUME A GAUSSIAN (or VON MISES) distribution & FIT all those variables under that ASSUMPTION
    #get orientations for each unit (in degrees)
    ori_ijs = decode.gettheta(np.arange(0,N),N) 
    
    # calculate 1) baseline activity
    # 2) amplitude of bump
    # 3) mean of bump
    # 4) standard deviation of bump
    #, assuming Gaussian (will NOT work if the stimulus is not N//2 (or close by))
    actb_fit = [curve_fit(decode.gauss1D_fit, ori_ijs, x, bounds=((0,0,-180,0),(np.inf,np.inf,180,180))) 
           for x in end_bumps]
    
    amplitudes[simii] = np.mean([actb_fit[x][0][1] for x in np.arange(0,nsims)])
    print('mean FIT amplitude of bump: ' + str(amplitudes[simii]))
    SDs[simii] = np.mean([actb_fit[x][0][3] for x in np.arange(0,nsims)])
    print('mean FIT standard deviation of bump: ' + str(SDs[simii]))
    
    

# ANALYZE SIMULATIONS COMPARATIVELY, nowthat key measures extracted from each
fig,ax = plt.subplots(nrows=1,ncols=3,figsize=(15.5,4))
ax[0].errorbar(np.arange(4),MSE,np.std(MSE)/np.sqrt(nsims))
ax[0].set_ylabel('Decoding error',fontsize=14)
ax[0].set_xlabel('Simulation')
ax[0].set_xticks([0,1,2,3])
ax[0].set_xticklabels(['0/0','0/1','1/0','1/1'])

ax[1].errorbar(np.arange(4),SDs,np.std(SDs)/np.sqrt(nsims))
ax[1].set_ylabel('Bump width',fontsize=14)
ax[1].set_xlabel('Simulation')
ax[1].set_xticks([0,1,2,3])
ax[1].set_xticklabels(['0/0','0/1','1/0','1/1'])

ax[2].errorbar(np.arange(4),amplitudes,np.std(amplitudes)/np.sqrt(nsims))
ax[2].set_ylabel('Bump amplitude',fontsize=14)
ax[2].set_xlabel('Simulation')
ax[2].set_xticks([0,1,2,3])
ax[2].set_xticklabels(['0/0','0/1','1/0','1/1'])


