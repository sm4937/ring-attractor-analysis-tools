#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thur May 27th, 2024

author: @sarah
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
from mpl_toolkits.axes_grid1 import make_axes_locatable


# Want to test under same starting conditions each time? (good to replicate
# when you're testing new ideas, bad if you want to run empirical simulations)
setSeed = True
nsims = 1

# Do you want a printout which includes the current and noise variables as well?
full_mat = False
unit_dist = False
categorical = False
apply_TMS = False #"TMS" effect (uniform white noise)


# IS ATTENTION BEING APPLIED TO THE STIMULUS LOCATION?
attention_flag = True
effort_flag = True
inhib_gain_flag = True

# How should effort be applied: at specific timepoints, or throughout trial?
# ("temporal" vs. "constant")
gain_type = 'temporal'


rng = 1
    
#set params
N = 360 # in Engel work this is 256

dfunc = simu.degdiff
dfunc2 = dfunc

# set connectivity params #

# connectivity offset (ie negative connections for distantly tuned)
# in Engel's papers, this value is 0.5
Jmn = 0.5 
#connectivity amplitude - what is amp in published work?
Jpp = 2.5 #3.0 
# in Engel 2011 paper this value is 2.2
#connectivity sd
sigmap = 43.2 #32

# can also set parameters for inhibitory cells:
#Jpn = 1.0
#sigman = 40

#create connectivity matrix        
gk1k2 = simu.gen_conn_tune(N,Jpp=Jpp,sigmap=sigmap,Jmn=Jmn,dfunc=dfunc)
   
#noise params
simu.sigma_n = 0.05 #0.0125
simu.I0 = 0.3297 #0.335 (for no/low noise case) 

#do we model a TMS effect?
if apply_TMS:
    TMS_params = {'tstart': 2.0,
                   'tend': 2.05, 
                   'IstimOn': simu.I0*2.0,
                   }
    # if applying TMS, do so at second 2 (post-stimulus delay period)
else:
    TMS_params = None

    
# produce correct x-ticks (describing span of connectivity matrix)
targ_theta = np.transpose(np.array([-180.,-90,0,90,180],ndmin=2))
theta = decode.gettheta(np.arange(0,N),N)  
ticks = np.argmin(np.abs(theta-targ_theta),axis=1)
degticks = theta[ticks]
degticks_int = np.int32(degticks)


# this is not where timing variables are created (they are globals inside ring_1D2a)
# but I need them here to know the number of timepoints I'm working with
dt = 1e-3
tfinal = 6
n_timepoints = len(np.arange(0, tfinal-(dt),dt))

effort_range = attention_range = np.arange(1,2.6,0.1)
square_side_length = len(effort_range)
MSE_grid_gauss = np.zeros((square_side_length,square_side_length))
width_grid_gauss = np.zeros((square_side_length,square_side_length))
amplitude_grid_gauss = np.zeros((square_side_length,square_side_length))

MSE_grid_VM = np.zeros((square_side_length,square_side_length))
width_grid_VM = np.zeros((square_side_length,square_side_length))
amplitude_grid_VM = np.zeros((square_side_length,square_side_length))

row_i = -1

for attention_gain_amount in attention_range:
    
    row_i += 1
    column_j = -1
    
    for effort_gain_amount in effort_range:
        
        column_j += 1
    
        print('Filling grid position i ' + str(row_i) + ', j ' + str(column_j))
        
        # #
        
        # # RUN NETWORK SIMULATION # # 
        
        # Pass in # of cells to simulate (N), the connectivity matrix between cells (gk1k2),
        # and the input (Iexternal). Specify random seed (important for reasons that are
        # not immediately clear to me), whether to save the simulation results, 
        # whether to generate a full matrix of firing rates/currents/noise, or just
        # determine the response of the network on some behavioral task (via
        # MLE decoding)
        
        # #
                
        # Generate the external input (STIMULUS TO STORE)
        # Set input parameters/settings
        
        #input settings
        inpPeak = N//2 #location of peak stimulus drive (directed to network center)
        inpSigma = sigmap
        
        gs = 0.02*4 #input amplitude parameter
        if attention_flag:
            gs *= np.round(attention_gain_amount,decimals=1)
            # apply contrast gain to stimulus/input
                
        # It's a 1-D gaussian centered on inpPeak, with sigma (at this time)
        # equal to the connectivity sigma (sigmap = 32)
        Iexternal = simu.stimuli(inpPeak, N,sigma=inpSigma,Jp=gs,dfunc=dfunc2)
        
        #set numerator gain (MULTIPLICATIVE)
        gain_amount = 1
        extra_inhib_amount = 0
        if effort_flag:
            gain_amount = np.round(effort_gain_amount,decimals=1) #2, 1.25
            # apply response gain to entire network 
            if inhib_gain_flag:
                extra_inhib_amount = 0.00055556
                # equivalent to increasing Jmn from 0.5 to 0.7
        
        
        # default values for the multiplicative gain, and subtractive inhibition
        gain_t = np.ones((n_timepoints+1,1))
        extra_inhib_t = np.zeros((n_timepoints+1,1))
        
        # is 
        if gain_type=='constant':
            # transform gain constant into time-dependent vector of equivalent values all over
            gain_t *= gain_amount
            extra_inhib_t += extra_inhib_amount
        elif gain_type== 'temporal':
            # when does onset on gain happen? let's say last 1.5 seconds
            duration = 1.5
            #tfinal_effort = tfinal
            tfinal_effort = tfinal - 4
            indices = range(int(tfinal_effort*(1/dt)-duration*(1/dt)),int(tfinal_effort*(1/dt)),1)
            gain_t[indices] *= gain_amount
            # extra inhibition is applied in a temporal fashion, too
            extra_inhib_t[indices] = extra_inhib_amount
        
                
        # initialize variable for saving decoding results from the end of the delay
        # period (akin to a behavioral response)
        
        # end_decoding = np.zeros((nsims,1))
        # end_decoding_pos = np.zeros((nsims,1))
        peakdegs = np.zeros(nsims)
        end_bumps = np.zeros((nsims,N))
        
        for simii in np.arange(nsims):
            
            if nsims > 1:
                print('Running simulation #' + str(simii))
                
            results=simu.simulation(N,gain_t,gk1k2,extra_inhib_t,Iexternal,rng=1,saveFile=False,
                                    setSeed=setSeed,TMS_params=TMS_params,full_mat=full_mat)
            # This simulation function lives inside ring_1D2a.py (inside this folder of code)
            
            # reshape outputs if the currents & noise have been concatenated (full_mat is set to true)
            if full_mat:
                currents = results[1*N : 2*N,:]
                noise = results[2*N : 3*N,:]
                results = results[0*N : 1*N,:]
            
            if categorical:
                thetafunc = decode.gettheta_cat
            else:
                thetafunc = decode.gettheta
        
            theta = thetafunc(np.arange(0,N),N)
            peakdeg = theta[inpPeak]
            peakdegs[simii] = peakdeg
            
            # decode the peak of the bump
            # decoding is decoding error
            # decoding = np.apply_along_axis(lambda x: decode.decoding1D(x,inpPeak,ori=theta), 0, results)
            # decoding_pos is the decoded location
            # decoding_pos = np.apply_along_axis(lambda x: decode.decoding1D(x,ori=theta), 0, results)
            
            # end_decoding[simii] = decoding[-1]
            # end_decoding_pos[simii] = decoding_pos[-1]
            
            # actually obtain the late delay period bump in order to also get its 
            # amplitude/gain 
            end_bumps[simii,:] = np.mean(results[:,5950:6000],axis=1)
            
        
        
        # PRODUCE OUTCOME MEASURES
        
        # ASSUME A GAUSSIAN (or VON MISES) distribution & FIT all those variables under that ASSUMPTION
        #get orientations for each unit (in degrees)
        ori_ijs = decode.gettheta(np.arange(0,N),N) 
        
        # calculate 1) baseline activity
        # 2) amplitude of bump
        # 3) mean of bump
        # 4) standard deviation of bump
        #, assuming Gaussian (will NOT work if the stimulus is not N//2 (or close by))
        actb_fit = [curve_fit(decode.gauss1D_fit, ori_ijs, x, bounds=((0,0,-180,0),(np.inf,np.inf,180,180))) 
               for x in end_bumps]
        
        end_baselines = [actb_fit[x][0][0] for x in np.arange(0,nsims)]
        end_amplitudes = [actb_fit[x][0][1] for x in np.arange(0,nsims)]
        end_means = [actb_fit[x][0][2] for x in np.arange(0,nsims)]
        end_SDs = [actb_fit[x][0][3] for x in np.arange(0,nsims)]

        # calculate absolute decoding errors
        decoding_errors = np.abs(end_means-peakdegs)

        # calculate mean squared error across all sims
        MSE = np.mean((end_means-peakdegs)**2)

        MSE_grid_gauss[row_i,column_j] = MSE
        width_grid_gauss[row_i,column_j] = np.mean(end_SDs)
        amplitude_grid_gauss[row_i,column_j] = np.mean(end_amplitudes)
        

        # equation for a von Mises is:
        #   p(x) = e^(kappa * cos(x - mu)) / 2 * pi * I0(kappa)
        #   where I0 is the modified Bessel function of order 0
        
        # set optimization problem up a little

        nparams = 4
        p0 = np.random.rand(nparams)

        # default p0 is all 1
        # default maxfev is 200*nparams

        actb_fit = [curve_fit(decode.vonMises_1D_fit, np.deg2rad(ori_ijs), x, p0=p0,
                              bounds=((0,0,-np.pi,0),(np.inf,300,np.pi,2*np.pi)),
                              maxfev=500*nparams) 
               for x in end_bumps]


        end_baselines = [actb_fit[x][0][0] for x in np.arange(0,nsims)]
        end_amplitudes = [actb_fit[x][0][1] for x in np.arange(0,nsims)]
        end_means = [actb_fit[x][0][2] for x in np.arange(0,nsims)]
        end_kappas = [actb_fit[x][0][3] for x in np.arange(0,nsims)]

        # Calculate FWHM (akin to an S.D. from V.M. concentration parameter Kappa)
        # Described in Chang et al., 2012, which cited Elstrott et al., 2008.

        FWHMs = [2 * np.arccos(np.log(0.5 * (np.exp(k) + np.exp(-k)))/k)
                 for k in end_kappas]
        
        MSE = np.mean((end_means-peakdegs)**2)
        
        MSE_grid_VM[row_i,column_j] = MSE
        width_grid_VM[row_i,column_j] = np.mean(FWHMs)
        amplitude_grid_VM[row_i,column_j] = np.mean(end_amplitudes)

        

# plot grids here!

# ANALYZE SIMULATIONS COMPARATIVELY, nowthat key measures extracted from each

# FIRST FROM GAUSSIAN DECODER
# plot MSE
fig,ax = plt.subplots(nrows=3,ncols=1,figsize=(7,18))
divider = make_axes_locatable(ax[0])
cax = divider.append_axes('right', size='5%', pad=0.05)
im = ax[0].imshow(MSE_grid_gauss, interpolation='nearest')
fig.colorbar(im, cax=cax, orientation='vertical')
ax[0].set_ylabel('Attention gain',fontsize=14)
ax[0].set_xlabel('Effort gain',fontsize=14)
# why are there 0 plotting languages which enable the following functionality easily? yeesh!
ax[0].set_xticks(np.arange(1,len(effort_range)+1),np.round(effort_range,decimals=1))
ax[0].set_yticks(np.arange(1,len(attention_range)+1),np.round(attention_range,decimals=1))
ax[0].set_title('MSE',fontsize=20)

# plot bump SD
divider = make_axes_locatable(ax[1])
cax = divider.append_axes('right', size='5%', pad=0.05)
im = ax[1].imshow(width_grid_gauss, interpolation='nearest')
fig.colorbar(im, cax=cax, orientation='vertical')
ax[1].set_ylabel('Attention gain',fontsize=14)
ax[1].set_xlabel('Effort gain',fontsize=14)
# why are there 0 plotting languages which enable the following functionality easily? yeesh!
ax[1].set_xticks(np.arange(1,len(effort_range)+1),np.round(effort_range,decimals=1))
ax[1].set_yticks(np.arange(1,len(attention_range)+1),np.round(attention_range,decimals=1))
ax[1].set_title('bump SD',fontsize=20)

# plot bump amplitude
divider = make_axes_locatable(ax[2])
cax = divider.append_axes('right', size='5%', pad=0.05)
im = ax[2].imshow(amplitude_grid_gauss, interpolation='nearest')
fig.colorbar(im, cax=cax, orientation='vertical')
ax[2].set_ylabel('Attention gain',fontsize=14)
ax[2].set_xlabel('Effort gain',fontsize=14)
# why are there 0 plotting languages which enable the following functionality easily? yeesh!
ax[2].set_xticks(np.arange(1,len(effort_range)+1),np.round(effort_range,decimals=1))
ax[2].set_yticks(np.arange(1,len(attention_range)+1),np.round(attention_range,decimals=1))
ax[2].set_title('bump amplitude',fontsize=20)
plt.show()


# NEXT FROM VON MISES DECODER
# plot MSE
fig,ax = plt.subplots(nrows=3,ncols=1,figsize=(7,18))
divider = make_axes_locatable(ax[0])
cax = divider.append_axes('right', size='5%', pad=0.05)
im = ax[0].imshow(MSE_grid_VM, interpolation='nearest')
fig.colorbar(im, cax=cax, orientation='vertical')
ax[0].set_ylabel('Attention gain',fontsize=14)
ax[0].set_xlabel('Effort gain',fontsize=14)
# why are there 0 plotting languages which enable the following functionality easily? yeesh!
ax[0].set_xticks(np.arange(1,len(effort_range)+1),np.round(effort_range,decimals=1))
ax[0].set_yticks(np.arange(1,len(attention_range)+1),np.round(attention_range,decimals=1))
ax[0].set_title('MSE',fontsize=20)

# plot bump SD
divider = make_axes_locatable(ax[1])
cax = divider.append_axes('right', size='5%', pad=0.05)
im = ax[1].imshow(width_grid_VM, interpolation='nearest')
fig.colorbar(im, cax=cax, orientation='vertical')
ax[1].set_ylabel('Attention gain',fontsize=14)
ax[1].set_xlabel('Effort gain',fontsize=14)
# why are there 0 plotting languages which enable the following functionality easily? yeesh!
ax[1].set_xticks(np.arange(1,len(effort_range)+1),np.round(effort_range,decimals=1))
ax[1].set_yticks(np.arange(1,len(attention_range)+1),np.round(attention_range,decimals=1))
ax[1].set_title('bump FWHM',fontsize=20)

# plot bump amplitude
divider = make_axes_locatable(ax[2])
cax = divider.append_axes('right', size='5%', pad=0.05)
im = ax[2].imshow(amplitude_grid_VM, interpolation='nearest')
fig.colorbar(im, cax=cax, orientation='vertical')
ax[2].set_ylabel('Attention gain',fontsize=14)
ax[2].set_xlabel('Effort gain',fontsize=14)
# why are there 0 plotting languages which enable the following functionality easily? yeesh!
ax[2].set_xticks(np.arange(1,len(effort_range)+1),np.round(effort_range,decimals=1))
ax[2].set_yticks(np.arange(1,len(attention_range)+1),np.round(attention_range,decimals=1))
ax[2].set_title('bump amplitude',fontsize=20)
plt.show()

# 


