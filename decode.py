#formerly Analyz.py (see there / Analyz_bak for old/testing code)

from __future__ import division
#from concurrent.futures import ProcessPoolExecutor
from scipy.optimize import curve_fit
from scipy.special import iv, i0, j0
import numpy as np
import pandas as pd
import glob
import re
import warnings
import os.path


def gettheta(N1,N):
    """returns the tuning angle for a given location in the network in degrees [-180,180)"""
    rawdiff = N1 - N/2
    absdiff = np.abs(rawdiff)
    return np.sign(rawdiff) * np.minimum(absdiff, N - absdiff) * 360 / N

def gettheta_cat(N1,N,baseN=20):
    assert (N % baseN)==0
    ori_ijs = gettheta(np.arange(0,baseN),baseN)
    ori_ijs = np.repeat(ori_ijs,N//baseN)
    return ori_ijs[N1]
   
def decoding(firing,N,ijcenter=None,outtype='split'):
    """uses population vector method (circular weighted mean) to decode
        see e.g. Seung & Sompolinksy, 1993
        
        Input: 
            firing: unit firing rates as 2D matrix or 1D vector
            N: network size is N**2
            ijcener (optional): ij (i.e. matrix) coordinates of stimulus peak: 2 element vector
            outtype: 'split' or 'deg'
            
        Output: 
            ijcenter=None: decoded position
            ijcenter!=None: decoding error from stimulus peak (raw error)
            outtype='split,' returns a 2-element vector of the decoded xy position/error (x,y)
            outtype='deg,' returns the Euclidean squared error (ijcenter must be specified)
        
    """
    
    if firing.shape==(N,N):
        fr = firing.reshape(N*N,1)
    elif firing.shape==(N*N,):
        fr = np.expand_dims(firing,1)
    
    assert fr.shape==(N*N,1)
    

    #get coding (in degrees) (same for each axis)
    ori_ijs = gettheta(np.arange(0,N),N) 
    ori_ijs = np.exp(1j*np.deg2rad(ori_ijs)) #convert degrees to complex rep
    
    #using tile and repeat to avoid nested loops/list comps 
    #(i.e outer loop over rows i, inner loop over columns j)
    ori_i = np.repeat(ori_ijs,N)
    ori_j = np.tile(ori_ijs,N)
    ori = np.column_stack((ori_i,ori_j))
    
    dcdVct = np.sum(fr*ori,axis=0) #computed weighted sum (population vector)
    
    #if passed a target (in ij coordinates), return error from target
    if ijcenter is not None:
        #if passed a tuple, we need it to be a list for numpy indexing
        dcdVct /= ori_ijs[list(ijcenter)] #counter-rotation is division in complex space
    
    dcdVct = np.angle(dcdVct,deg=True) #return to degrees
    
    if outtype == 'split':
        #convert to xy coordinates [the flip of ij coordinates]
        #flipping so x,y is 0,1. This is the general confusion caused by rows being the
        #vertical dimension (y) of an image, but being the first dimension of a matrix.
        return np.flip(dcdVct) 
    elif outtype=='deg':
        assert ijcenter is not None
        return np.sqrt(dcdVct[0] ** 2 + dcdVct[1] ** 2)
    else:
        raise ValueError('Invalid outtype')
        
        
def decoding_N_spread1D(firing,baseN=20,**kwargs):
    """Given a firing rate vector, find the units corresponding to a smaller 
    network of units with equally-spaced tuning of size baseN
    
    """
    assert firing.ndim==1 #only deal w/ vectors at this point
    N = firing.size 
    assert N >= baseN
    
    #get spacing of smaller network
    orib = gettheta(np.arange(0,baseN),baseN)

    #get spacing of current network
    ori = gettheta(np.arange(0,N),N)

    #find indicies of units in the larger network that are the closest match for 
    #the tuning in the smaller network
    ori_eq = np.abs(ori[np.newaxis,:]-orib[:,np.newaxis])
    midx = np.argmin(ori_eq,axis=1)
    
    kwargs['ori'] = ori[midx]
    
    return decoding1D(firing[midx],**kwargs) #do decoding on network subset

    
    
def decoding1D(firing,icenter=None,ori=None):
    """uses population vector method (circular weighted mean) to decode
        see e.g. Seung & Sompolinksy, 1993
        one could imagine making this more robust and allowing to center on arbitrary
        angle, but since network input does not allow that not allowed here either
    """
   
     
    assert firing.ndim==1 #only deal w/ vectors at this point
    N = firing.size 
   
    #compute differences in degrees between each unit and stim location
    if ori is None:
        ori = gettheta(np.arange(0,N),N)
    
    ori = np.exp(1j*np.deg2rad(ori)) #convert degrees to complex rep

    dcdVct = np.dot(firing,ori) #computed weighted sum (population vector)
    
    #if passed a target (in ij coordinates), return error from target
    if icenter is not None:
        dcdVct /= ori[icenter] #counter-rotation is division in complex space
    
    dcdVct = np.angle(dcdVct,deg=True) #return to degrees
    
    return dcdVct
        
def do_decoding(in_file,decoder=decoding1D,**kwargs):
    """
        given an input file (pattern), loads the associated activity
    """
    
    sim_files = glob.glob(in_file)
    
    rngpat = 'rng(\d{2,4})'

        
    decodedf = []
    for i,f in enumerate(sim_files):
        sim = np.load(f)
        
        #get network size
        N = sim.shape[0]
        
        decoding = np.apply_along_axis(lambda x: decoder(x,**kwargs), 0, sim)
        
        ddf = {'N':N,'decpos':decoding}
        
        rng = re.search(rngpat,f)
        
        if rng is not None:
            ddf['rng'] = int(rng.group(1))
        else:
            ddf['rng'] = i 
            warnings.warn('Could not extract seed from file name. Numbering in order of file processing. Check data carefully.')
        
        ddf = pd.DataFrame(ddf)
        decodedf.append(ddf)    
        
    decodedf = pd.concat(decodedf)
    decodedf.index.rename('time',inplace=True)
    decodedf.set_index(['N','rng'],append=True,inplace=True )
    decodedf.sort_index(level=['N','rng','time'],inplace=True)
    
    return decodedf


# def decoding_p_helper(f,decoder,rngpat,**kwargs):
#     sim = np.load(f)
    
#     #get network size
#     N = sim.shape[0]
    
#     decoding = np.apply_along_axis(lambda x: decoder(x,**kwargs), 0, sim)
    
#     ddf = {'N':N,'decpos':decoding}
    
#     rng = re.search(rngpat,f)

#     ddf['rng'] = int(rng.group(1))
    
#     return pd.DataFrame(ddf)


# def do_decoding_p(in_file,decoder=decoding1D,**kwargs):
#     """
#         given an input file (pattern), loads the associated activity
#     """
    
#     sim_files = glob.glob(in_file)
    
#     rngpat = 'rng(\d{2,4})'
    
#     decode_func = lambda x:decoding_p_helper(x,decoder,rngpat,**kwargs)
        
#     with ProcessPoolExecutor(max_workers=4) as executor:
#         decodedf = executor.map(decode_func,sim_files)
        
#     decodedf = pd.concat(decodedf)
#     decodedf.index.rename('time',inplace=True)
#     decodedf.set_index(['N','rng'],append=True,inplace=True )
#     decodedf.sort_index(level=['N','rng','time'],inplace=True)
    
#     return decodedf


def load_decoding(indir,infile_pat,verbose=True):
    decfiles = glob.glob(os.path.join(indir,infile_pat))
    if verbose:
        print(decfiles)     
    decdf = [pd.read_csv(f) for f in decfiles]
    decdf = pd.concat(decdf)
    decdf.set_index(['N','rng','time'],inplace=True)
    decdf.sort_values(['N','rng','time'],inplace=True)
    
    #compute absolute decoding error
    decdf['adecpos'] = decdf['decpos'].abs()
    
    return decdf


#for fitting a Gaussian to the delay-period bump
def gauss1D_fit(x, Jm, Jp, mu, sigma):
    return Jm + Jp * np.exp(-(x-mu) ** 2 / (2 * sigma ** 2))

#for fitting a von mises (circular gaussian) to the delay-period bump
def vonMises_1D_fit(x, Jm, Jp, mu, kappa):
    
    # VM = 0 + 1 * ((np.exp(kappa * np.cos(np.linspace(-np.pi,np.pi,360) - mu))) / (2 * np.pi * iv(0,kappa)))
    base_VM = ((np.exp(kappa * np.cos(x-mu))) / (2 * np.pi * iv(0,kappa)))
    base_VM /= np.max(base_VM)
    return Jm + Jp * base_VM
    # scipi.special.i0 is modified Bessel function of order 0 (e.g. i0(kappa))
    # scipy.special.iv is modified bessel function of first kind (args[0] = order, args[1] = argument)
    
    # base_VM = ((np.exp(kappa * np.cos(x-mu))) / (2 * np.pi * i0(kappa)))
    # normalized_VM = base_VM/np.max(base_VM)
    # return Jm + Jp * normalized_VM


def fit_bump_bin(act,tbin=50,tstart=1001,thetafunc=gettheta):
    '''
    This function is used to fit a gaussian to determine bump width. Fit to mean
    activity in non-overlapping time bins.
    
    
    Parameters
    ----------
    act : n units x t time points activity matrix.
    tbin : int, optional
        width (in ms) of time bins. May not be quite equal depending on actual number of time points
        The default is 50.
    tstart : int, optional
        time (in ms) to begin fitting. The default is 1001.
    thetafunc : function, optional
        function to use to label neuron tuning. The default is gettheta.

    Returns
    -------
    list
        DESCRIPTION.

    '''
    
    #get orientations for each unit (in degrees)
    ori_ijs = thetafunc(np.arange(0,act.shape[0]),act.shape[0]) 
    
    #restrict fitting to start time specified (currently not supporting end time)
    fittimes = np.arange(tstart,act.shape[1])
    act_fit = act[:,tstart:]
    
    #commence extracting bins and labeling time points by bin
    bin_edges = np.histogram_bin_edges(fittimes,bins=len(fittimes)//tbin)
    binNs = np.digitize(fittimes, bin_edges)
    binNs[-1] = binNs[-2] #include last value in last bin...why isn't this the default???
    bin_nums = np.unique(binNs)

    
    #compute mean in each bin and fit
    actb_mean = [np.mean(act_fit[:,binNs==b],axis=1) for b in bin_nums]
    actb_fit = [curve_fit(gauss1D_fit, ori_ijs, x, bounds=((0,0,-180,0),(np.inf,np.inf,180,180))) 
            for x in actb_mean]
    
    #compute time bin centers
    fittimes_b = [np.median(fittimes[binNs==b]) for b in bin_nums]
    
    #return 
    return actb_fit,fittimes_b,actb_mean,bin_edges



if __name__=='__main__':
    #YOU NEED TO UPDATE THIS!!
    
    N = 10
    Ntot = N**2
    
    simN = 100
    fn = 'decodeAvg_N_'+"{0:.0f}".format(N)+'.txt'
    directory = './' #'/Users/Allen/Documents/Wanglab/ClayWM/rateRing2D/'
    print(directory+fn)
    tc = np.loadtxt(directory+fn)
    resultName = 'decodePos_'+"{0:.0f}".format(N)+'.txt'
    
    # num = np.arange(len(tc[0]))
    # print num
    print(tc.shape)
    dcdPos = np.ones(simN)
    dcdPos_Amp = np.ones(simN)
    
    
    ArrErrordeg = []
    for j in range(simN):
        tc_now = tc[j]
        tc2d_now =tc_now.reshape((N,N))
        id, jd = decoding(tc2d_now, N/2, N/2, N)
        errordeg = np.sqrt( (id*360/N)**2 + (jd*360/N)**2 )
        ArrErrordeg.append(errordeg)
        # smooth_now = smooth2d_now.reshape((Ntot,))
        # dcdPos[j] = np.argmax(smooth_now)
        # dcdPos_Amp[j] = np.amax(smooth_now)
    
    np.savetxt(directory+resultName, ArrErrordeg, fmt='%.5f')