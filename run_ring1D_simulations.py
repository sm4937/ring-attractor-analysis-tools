#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thur March 21st, 2024

Original code from XJ Wang's group
Adapted for our purposes by Nathan Tardiff
Further adapted to test effects of effort in simulations by Sarah Master

This supercedes ring1D_testing and ring1D_testing_dist
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.stats as st
import pandas as pd
#import glob
import os.path
import datetime
import scipy
from scipy.optimize import curve_fit

# not public code, use by permission of XJ Wang only:
import ring_1D2a as simu
import decode



#set up simulation global vars (don't love this, but OK)

#NOTE: Brunel & Wang, 2001 use 18 degrees for connectivity footprint
#Wimmer et al., 2014 use ~36.4 degrees (von mises w/ kappa==1.5), and same for stim.

# def sqrt_N_scaling(gk1k2,N,**kwargs):
#     #scaling is by N, per Compte et al., 2000
#     return gk1k2 / np.sqrt(N)


# def make_cat_unbalanced2k():
    
#         unit_Ns_left = np.arange(0,7)
#         unit_Ns_left = np.repeat(unit_Ns_left,128)
        
#         unit_Ns_center = np.arange(7,14)
#         unit_Ns_center = np.repeat(unit_Ns_center,48)
        
            
#         unit_Ns_right = np.arange(14,20)
#         unit_Ns_right = np.repeat(unit_Ns_right,128)
        
#         unit_Ns = np.concatenate((unit_Ns_left,unit_Ns_center,unit_Ns_right))
        
#         assert(len(unit_Ns)==2000)
        
#         return unit_Ns
        
        

# Want to run a new simulation? (not preload old results?)
new_sim = True

# Want to test under same starting conditions each time? (good to replicate
# when you're testing new ideas, bad if you want to run empirical simulations)
setSeed = False
nsims = 480

# Do you want a printout which includes the current and noise variables as well?
full_mat = True


unit_dist = False
categorical = False

apply_TMS = False #"TMS" effect (uniform white noise)



rng = 1

if new_sim:
    
    #set params
    N = 360 # in Engel work this is 256
    
    if categorical:
        # unit_Ns = make_cat_unbalanced2k()
        # dfunc = lambda N1,N2,N: simu.degdiff_cat(N1,N2,N,unit_Ns=unit_Ns) 
        dfunc = simu.degdiff_cat
        dfunc2 = dfunc
    elif unit_dist:
        dfunc = simu.degdiff400
        dfunc2 = simu.degdiff
    else:
        dfunc = simu.degdiff
        dfunc2 = dfunc
    
    
    #simu.Ntot = N
    
    # SARAH: SIMS USUALLY *NOT* UNIT DIST
    if not unit_dist:
        #connectivity params    
        
        Jmn = 0.5 #connectivity offset (ie negative connections for distantly tuned)
        # in Engel's papers, this value is 0.5
        
        Jpp = 2.5 #3.0 #connectivity amplitude (in Engel it's 2.2 but that doesn't work here)
        sigmap = 43.2 #32 #connectivity sd
        
        # can also set parameters for inhibitory cells:
        #Jpn = 1.0
        #sigman = 40

        #create connectivity matrix        
        gk1k2 = simu.gen_conn_tune(N,Jpp=Jpp,sigmap=sigmap,Jmn=Jmn,dfunc=dfunc)
        
        # JPN IS NOT CURRENTLY AN ARGUMENT I'M PASSING IN, BUT I WILL NEED TO DO SO IF I'M GOING TO ENHANCE INHIBITION VIA GAIN, TOO
        # What happens when I set Jpn = Jpp?
        
    else:
        #trying to better match standage
        
        #excite conn profile
        sigmap = 32 #11.46
        
        #determine scale factor for matching unit conn strength (J) to tuning conn strength
        #at matches network (N=400)
        #connsum = np.sum(simu.gauss1D(np.arange(0,400),0,Jm=0,Jp=1,sigma=sigmap,N=400))
        #Jpscale = np.sqrt(connsum)/400
        
        Jpp = 3.0 #in tuning scaled units #8.25 #6.0 
        Jpp = simu.J_tune_to_unit(Jpp) #,sigmap)
        #Jpp *= Jpscale #scale factor
        #Jm = 0 #-0.1 #-0.5 #/(N/400) #-0.5

        #inhib connectivity profile
        sigman = 22.92 #180#22.92
        Jpn = 0 #0.5 #1.75
        Jmn = 0.546 #30 #0
        
        
        #generate connectivity matrix
        gk1k2 = simu.gen_conn_unit(N,Jpp=Jpp,sigmap=sigmap,Jmn=Jmn,Jpn=Jpn,
                                   sigman=sigman,dfunc=dfunc)
        
        
    
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
    
    with mpl.rc_context({'lines.linewidth': 2,
                    'font.family': 'Arial',
                    'font.size': 16,
                    'axes.spines.top': False,
                    'axes.spines.right': False,
                     'legend.frameon': False,
                     'legend.fontsize': 14,
                     'legend.markerscale': 0.0,
                    }):
    

        # plot connectivity matrix (gk1k2)  
        fig,ax=plt.subplots(nrows=1,ncols=4,figsize=(20,4.5))
        ax[0].imshow(gk1k2,cmap='RdBu')
        ax[0].set_xticks(ticks,labels=degticks_int)
        ax[0].set_yticks(ticks,labels=degticks_int)
        ax[0].axvline(N//2,color='k',alpha=0.4,linestyle='--')
        #plt.colorbar()
        
        
        # plot a slice of the connectivity matrix at cell 0 (tuned to 0 degrees)
        ax[1].plot(gk1k2[N//2,:],color='k')
        ax[1].set_xticks(ticks,labels=degticks_int)
        #ax[1].set_yticks([0,np.max(gk1k2[N//2,:])])
        ax[1].set_yticks([0,np.max(gk1k2[N//2,:])],
                         labels=[0,np.round(np.max(gk1k2[N//2,:])*100,1)])
        #ax[1].ticklabel_format(style='sci',axis='y')
        ax[1].axhline(0,color='k',alpha=0.5,linestyle='--')
        ax[1].axvline(N//2,color='k',alpha=0.5,linestyle='--')
        ax[1].set_ylabel('connectivity strength (au)')
        ax[1].set_title('central unit')
        
        # plot to demonstrate the circularity of the network: edge cells are still
        # connected to nearby edge cells, (e.g. cells tuned to 359 & 0 degrees)
        ax[2].plot(gk1k2[0,:],color='k')
        ax[2].set_xticks(ticks,labels=degticks_int)
        ax[2].set_yticks([0,np.max(gk1k2[N//2,:])],
                         labels=[0,np.round(np.max(gk1k2[N//2,:])*100,1)])
        ax[2].axhline(0,color='k',alpha=0.5,linestyle='--')
        ax[2].axvline(N//2,color='k',alpha=0.5,linestyle='--')
        ax[2].set_title('edge unit')
                
        #fig.tight_layout()
        
        fig.supxlabel('tuning (deg)',fontsize=16)
        
        I = np.arange(0.1,1,step=0.01)
        # R = simu.I2f(I)
        
        a = 270 #nA/hz
        b = 108 #Hz
        d = 0.154 #seconds
        
        # default gain parameter (a) is 270 nA/hz
        R = (a * I - b) / (1 - np.exp(-d * (a * I - b)))
        ax[3].plot(I,R)
        R2 = 1.25*(a * I - b) / (1 - np.exp(-d * (a * I - b)))
        ax[3].plot(I,R2)
        R3 = 1.5*(a * I - b) / (1 - np.exp(-d * (a * I - b)))
        ax[3].plot(I,R3)
        R4 = 2*(a * I - b) / (1 - np.exp(-d * (a * I - b)))
        ax[3].plot(I,R4)
        ax[3].set_title('I-O function')
        ax[3].legend(['gain = 1', 'gain = 1.25', 'gain = 1.5', 'gain = 2'])


    #generate connectivity and input
    #simu.gk1k2 = gk1k2
    #simu.Iexternal = Iexternal
    
    # this is not where timing variables are created (they are globals inside ring_1D2a)
    # but I need them here to know the number of timepoints I'm working with
    dt = 1e-3
    tfinal = 6
    n_timepoints = len(np.arange(0, tfinal-(dt),dt))
    
    # # RUN NETWORK SIMULATION # # 
    
    # Pass in # of cells to simulate (N), the connectivity matrix between cells (gk1k2),
    # and the input (Iexternal). Specify random seed (important for reasons that are
    # not immediately clear to me), whether to save the simulation results, 
    # whether to generate a full matrix of firing rates/currents/noise, or just
    # determine the response of the network on some behavioral task (via
    # MLE decoding)
    
    # IS THIS AN EFFORTFUL TRIAL? 
    effort_flag = True
    inhib_gain_flag = True
    
    # How should effort be applied: at specific timepoints, or throughout trial?
    # ("temporal" vs. "constant")
    gain_type = 'temporal'
    
    # IS ATTENTION BEING APPLIED TO THE STIMULUS LOCATION?
    attention_flag = False
    
    
    # Generate the external input (STIMULUS TO STORE)
    # Set input parameters/settings
    
    #input settings
    stim_list = np.random.randint(0,359,nsims)
    
    # set amount of numerator/denominator gain (default value = 270, effortful value = 275)
    #gain = 270; if effort_flag: gain = 275;
    
    #set numerator gain (MULTIPLICATIVE)
    gain_amount = 1
    extra_inhib_amount = 0
    if effort_flag:
        gain_amount = 2 #2
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
        tfinal_effort = tfinal-2
        indices = range(int(tfinal_effort*(1/dt)-duration*(1/dt)),int(tfinal_effort*(1/dt)),1)
        gain_t[indices] *= gain_amount
        # extra inhibition is applied in a temporal fashion, too
        extra_inhib_t[indices] = extra_inhib_amount

    
    # initialize variable for saving decoding results from the end of the delay
    # period (akin to a behavioral response)
    end_decoding = np.zeros((nsims,1))
    end_decoding_pos = np.zeros((nsims,1))
    end_bumps = np.zeros((nsims,N))
    peakdegs = np.zeros(nsims)
    
    for simii in np.arange(nsims):
        
        if nsims > 1:
            full_mat = False
            print('Running simulation #' + str(simii))
            
        inpPeak = stim_list[simii] # N//2 #location of peak stimulus drive
        inpSigma = sigmap 
        gs = 0.02*4 #input amplitude parameter
        
        if attention_flag:
            gs *= 2 #2
            # apply contrast gain to stimulus/input
                
        # It's a 1-D gaussian centered on inpPeak, with sigma (at this time)
        # equal to the connectivity sigma (sigmap = 32)
        Iexternal = simu.stimuli(inpPeak, N,sigma=inpSigma,Jp=gs,dfunc=dfunc2)
            
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

        # theta maps the input space (1 to 360) to the simulation space (-180 to 180)
        theta = thetafunc(np.arange(0,N),N)    
        
        # save the input stimulus's location (in simulation space)
        peakdeg = theta[inpPeak]
        peakdegs[simii] = peakdeg
        
        # decode the peak of the bump
        # decoding is decoding error
        decoding = np.apply_along_axis(lambda x: decode.decoding1D(x,inpPeak,ori=theta), 0, results)
        # decoding_pos is the decoded location
        decoding_pos = np.apply_along_axis(lambda x: decode.decoding1D(x,ori=theta), 0, results)
        
        end_decoding[simii] = decoding[-1]
        end_decoding_pos[simii] = decoding_pos[-1]
        
        # actually obtain the late delay period bump in order to also get its 
        # amplitude/gain 
        duration_of_avg = 50
        end_bumps[simii,:] = np.mean(results[:,-duration_of_avg:],axis=1)
        

        
# ELSE: DON'T WANT TO RUN A NEW SIMULATION!
else:
    sim_dir = './sims' #'/System/Volumes/Data/datc/nathan/net_sims/sims' #'./sims' #'/System/Volumes/Data/d/DATC/datc/nathan/net_sims/sims' #'./sims' #'/System/Volumes/Data/d/DATC/datc/nathan/net_sims/sims'
    
    test_file = 'All_act_1D_N400_tune0_stim2_2_rng1_R_2023-04-19.npy' #'All_act_1D_N800_unit0_rng1_2023-03-03.npy' #'All_act_1D_N20_rng100_2023-01-18.npy' #'All_act_N60_distdeg_rng1600_2022-11-01.npy'
    
    #this is hacky but best for handling legacy files, etc.
    test_currents = 'All_act_1D_N400_tune0_stim2_2_rng1_C_2023-04-19.npy' 
    
    test_noise = 'All_act_1D_N400_tune0_stim2_2_rng1_N_2023-04-19.npy' 
    
    print('Loading file: %s' % os.path.join(sim_dir,test_file))
    
    results = np.load(os.path.join(sim_dir,test_file))
    
    if full_mat:
        print('Loading currents file: %s' % os.path.join(sim_dir,test_currents))
        currents = np.load(os.path.join(sim_dir,test_currents))
        
        print('Loading noise file: %s' % os.path.join(sim_dir,test_noise))
        noise = np.load(os.path.join(sim_dir,test_noise))
    
    #get network size
    N = results.shape[0]
    
    #assuming activity in center of network. Would have to save this info if ever deviate from this
    inpPeak = N//2 



# ASSUME A GAUSSIAN (or VON MISES) distribution & FIT all those variables under that ASSUMPTION
#get orientations for each unit (in degrees)
ori_ijs = decode.gettheta(np.arange(0,N),N) 

# calculate 
# 1) baseline of bump of activity
# 2) amplitude of bump (peak)
# 3) mean of bump
# 4) standard deviation of bump
#, assuming Gaussian (will NOT work if the stimulus is not N//2 (or close by))
actb_fit = [curve_fit(decode.gauss1D_fit, ori_ijs, x, bounds=((0,0,-180,0),(np.inf,np.inf,180,180))) 
       for x in end_bumps]

end_means_gauss = [actb_fit[x][0][2] for x in np.arange(0,nsims)]
print('mean mean of GAUSSIAN bump: ' + str(np.mean(end_means_gauss)))
end_SDs = [actb_fit[x][0][3] for x in np.arange(0,nsims)]
print('mean standard deviation of GAUSSIAN bump: ' + str(np.mean(end_SDs)))

# equation for a von Mises is:
#   p(x) = e^(kappa * cos(x - mu)) / 2 * pi * I0(kappa)
#   where I0 is the modified Bessel function of order 0

# set optimization problem up a little

# nparams = 4
# p0 = np.random.rand(nparams)

# # default p0 is all 1
# # default maxfev is 200*nparams

# actb_fit = [curve_fit(decode.vonMises_1D_fit, np.deg2rad(ori_ijs), x, p0=p0,
#                       bounds=((0,0,-np.pi,0),(np.inf,np.inf,np.pi,2*np.pi)),
#                       maxfev=500*nparams) 
#        for x in end_bumps]

# end_baselines = [actb_fit[x][0][0] for x in np.arange(0,nsims)]
# print('mean baseline activation of VM bump: ' + str(np.mean(end_baselines)))
# end_amplitudes = [actb_fit[x][0][1] for x in np.arange(0,nsims)]
# print('mean amplitude of VM bump: ' + str(np.mean(end_amplitudes)))
# end_means = [actb_fit[x][0][2] for x in np.arange(0,nsims)]
# print('mean mean of VM bump: ' + str(np.mean(np.rad2deg(end_means))))
# end_kappas = [actb_fit[x][0][3] for x in np.arange(0,nsims)]

# Calculate FWHM (akin to an S.D. from V.M. concentration parameter Kappa)
# Described in Chang et al., 2012, which cited Elstrott et al., 2008.

# FWHMs = [2 * np.arccos(np.log(0.5 * (np.exp(k) + np.exp(-k)))/k)
#          for k in end_kappas]
# print('mean FWHM of VM bump: ' + str(np.mean(np.rad2deg(FWHMs))))



# calculate absolute decoding errors
decoding_errors = np.abs(end_means_gauss-peakdegs)

# calculate mean squared error across all sims
MSE = np.mean((end_means_gauss-peakdegs)**2)


if nsims > 1:
    
    # PLOT CORRECTNESS OVER NTRIALS
    fig,ax = plt.subplots()
    ax.hist(decoding_errors)
    ax.set_xlabel('Decoding error')
    
    # how much gain was applied?
    ax.set_title('Gain = ' + str(gain_amount) + ', MSE = ' + str(MSE)) 
    
    # SAVE OUTPUTS FOR COMPARISON BETWEEN CONDITIONS!
    if effort_flag:
        sim_filename = 'simulation_outputs/middle_of_delay_effort_sim'
        if inhib_gain_flag:
            sim_filename += '_with_inhibition'
    else:
        sim_filename = 'simulation_outputs/no_effort_sim'
        
    if attention_flag:
        sim_filename += '_with_attention'
    else:
        sim_filename += '_without_attention'
        
    stim_list_filename = sim_filename + '_stimuli.csv'
    sim_filename += '.csv'
    
    print(sim_filename)
    print(stim_list_filename)
    
    np.savetxt(sim_filename,end_bumps,delimiter=',')
    np.savetxt(stim_list_filename,stim_list,delimiter=',')


elif nsims == 1:
    
    # grab important variables for plotting time course of activity
    ticks = np.int32(np.linspace(0,N-1,11)) #np.arange(0,N,N//10)
    degticks = theta[ticks]
    #ticks = np.linspace(0,N,10)
    #degticks = np.int32(thetafunc(ticks,N))

    #times to plot decoding from
    dectimes = np.arange(500,simu.tfinal*1000-1) 
    alltimes = np.arange(0,len(decoding))

    #unit with tuning label closest to decoded angle
    dec2unit = np.fromiter((np.argmin(np.abs(theta-x)) for x in decoding),int,count=len(decoding))

    # # PLOT FIRING RATES AS A FUNCTION OF TIME AND PROXIMITY TO THE INPUT STIMULUS # #
    fig,ax = plt.subplots(figsize=(20,4.5))
    act=ax.imshow(results)
    ax.axhline(inpPeak,linestyle='--',linewidth=0.5,color='k',alpha=0.75)
    ax.plot(dectimes,dec2unit[dectimes],color='red',linewidth=0.5,alpha=.75)
    ax.set_aspect(results.shape[1]/(N*2))
    ax.set_yticks(ticks,labels=degticks)
    fig.colorbar(act)
    
    # PLOT CORRECTNESS OF SINGLE TRIAL
    fig,ax=plt.subplots()
    ticks = np.linspace(0,N,10)
    degticks = np.int32(decode.gettheta(ticks,N))
    # plot input stimulus (true stim in memory)
    ax.plot(Iexternal/np.max(Iexternal))
    ax.axvline(inpPeak,linestyle='--',linewidth=0.5,color='k',alpha=0.75)
    ax.set_xticks(ticks,labels=degticks)
    # compute mean late delay period firing rate
    # normalize to be in same range as input stimulus
    activity_bump = np.mean(results[:,5950:6000],axis=1)
    #ax.set_title('Max activity: ' + str(np.max(activity_bump)))
    
    if effort_flag:
        title_text = 'Effortful trial'
    else:
        title_text = 'Non-effortful trial'
        
    if attention_flag:
        title_text += ' with attention'
    else:
        title_text += ' without attention'
    ax.set_title(title_text)
    
    #activity_bump /= np.max(activity_bump)
    ax.plot(activity_bump)
    ax.legend(['external input to network','mean activity (late delay)'])
    