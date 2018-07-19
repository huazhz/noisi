from __future__ import print_function
from mpi4py import MPI
import numpy as np
import os
import h5py
import json
import click
from glob import glob
from math import ceil
import time
from scipy.signal.signaltools import fftconvolve
try:
    from scipy.fftpack import next_fast_len, hilbert
except ImportError:
    from noisi.borrowed_functions.scipy_next_fast_len import next_fast_len
from obspy import Trace, read, Stream
from noisi import NoiseSource, WaveField, BasisFunction
from noisi.util import geo
from obspy.signal.invsim import cosine_taper
from noisi.util import filter
try:
    from scipy.signal import sosfilt
except ImportError:
    from obspy.signal._sosfilt import _sosfilt as sosfilt
from noisi.util.windows import my_centered, zero_buddy
from noisi.util.geo import geograph_to_geocent
from noisi.util.corr_pairs import *
import matplotlib.pyplot as plt
import instaseis


class kernel_params(object):
    def __init__(self):
        pass

#ToDo: put in the possibility to run on mixed channel pairs
def paths_input(cp,source_conf,step,ignore_network,instaseis):
    
    inf1 = cp[0].split()
    inf2 = cp[1].split()
    
    conf = json.load(open(os.path.join(source_conf['project_path'],
        'config.json')))
    measr_conf = json.load(open(os.path.join(source_conf['source_path'],
        'measr_config.json')))
    channel = source_conf['channel']
    
    # station names
    if ignore_network:
        sta1 = "*.{}..{}".format(*(inf1[1:2]+[channel]))
        sta2 = "*.{}..{}".format(*(inf2[1:2]+[channel]))
    else:
        sta1 = "{}.{}..{}".format(*(inf1[0:2]+[channel]))
        sta2 = "{}.{}..{}".format(*(inf2[0:2]+[channel]))


    # Wavefield files  
    if instaseis == False:
        if source_conf['preprocess_do']:
            dir = os.path.join(source_conf['source_path'],
                'wavefield_processed')
            
        else:
            dir = conf['wavefield_path']
    
        wf1 = glob(os.path.join(dir,sta1+'.h5'))[0]
        wf2 = glob(os.path.join(dir,sta2+'.h5'))[0]
    else:
        # need to return two receiver coordinate pairs. 
        # For buried sensors, depth could be used but no elevation is possible,
        # so maybe keep everything at 0 m?
        # lists of information directly from the stations.txt file.
        wf1 = inf1
        wf2 = inf2

    
    # Starting model for the noise source
   
        # The base model contains no spatial or spectral weights.
    nsrc = os.path.join(source_conf['project_path'],
                 source_conf['source_name'],'step_'+str(step),
                 'base_model.h5')
  
    # Adjoint source
    # if measr_conf['mtype'] in ['energy_diff']:
    #     adj_src_basicnames = [ os.path.join(source_conf['source_path'],
    #              'step_'+str(step),
    #              'adjt',"{}--{}.c".format(sta1,sta2)),
    #              os.path.join(source_conf['source_path'],
    #              'step_'+str(step),
    #              'adjt',"{}--{}.a".format(sta1,sta2))]
    # else:
    adj_src_basicnames = os.path.join(source_conf['source_path'],
                 'step_'+str(step),
                 'adjt',"{}--{}".format(sta1,sta2))


     
    return(wf1,wf2,nsrc,adj_src_basicnames)
    
    
def get_ns(conf,source_conf,insta):
    
    # Nr of time steps in traces
    if insta:
        # get path to instaseis db
        #ToDo: ugly.
        dbpath = json.load(open(os.path.join(source_conf['project_path'],
            'config.json')))['wavefield_path']
        # open 
        db = instaseis.open_db(dbpath)
        # get a test seismogram to determine...
        stest = db.get_seismograms(source=instaseis.ForceSource(latitude=0.0,
            longitude=0.0),receiver=instaseis.Receiver(latitude=10.,
            longitude=0.0),dt=1./source_conf['sampling_rate'])[0]
        
        nt = stest.stats.npts
        Fs = stest.stats.sampling_rate
    else:
        if source_conf['preprocess_do']:
            wfs = glob(os.path.join(source_conf['source_path'],
            'wavefield_processed',
            '*.h5'))
        else:
            wfs = glob(os.path.join(conf['wavefield_path'],'*.h5'))

        try:
            wf1 = wfs[0]
        except IndexError:
            raise ValueError("No wavefield input found.")

        with WaveField(wf1) as wf1:
            nt = int(wf1.stats['nt'])
            Fs = round(wf1.stats['Fs'],8)
    
    # Necessary length of zero padding for carrying out 
    # frequency domain correlations/convolutions
    n = next_fast_len(2*nt-1)     
    
    # Number of time steps for synthetic correlation
    n_lag = int(source_conf['max_lag'] * Fs)
    if nt - 2*n_lag <= 0:
        click.secho('Resetting maximum lag to %g seconds: Synthetics are too\
 short for a maximum lag of %g seconds.' %(nt//2/Fs,n_lag/Fs))
        n_lag = nt // 2
        
    n_corr = 2*n_lag + 1
    
    return nt,n,n_corr,Fs
        
    
def get_nr_measurements(bandpass):
    
    if bandpass == None:
        filtcnt = 1
    elif type(bandpass) == list:
        if type(bandpass[0]) != list:
            filtcnt = 1
        else:
            filtcnt = len(bandpass)
    else:
        raise ValueError('Bandpass in measr_config.json\
 must be \'null\' or list.')

    return(filtcnt)


def get_adjoint_spectra(adjt,params):

    adjt_spect = np.zeros((params.filtcnt,params.n_freq),dtype=np.complex)
    
    for ix_f in range(params.filtcnt):
        adjtf = adjt + '*.{}.sac'.format(ix_f)
        adjtfile = glob(adjtf)

        try:
            f = read(adjtfile[0])[0]
            # transform the adjoint source to frequency domain:
            ix_mid = f.stats.npts // 2
            adjstf = np.zeros(params.n)
            adjstf[-ix_mid:] = f.data[0:ix_mid]
            adjstf[0:ix_mid+1] = f.data[ix_mid:]
            adjspc = np.fft.rfft(adjstf,n=params.n)
            adjspc[1:] *= 2. # correct amplitude for RFFT 
            adjt_spect[ix_f,:] = np.conjugate(
                adjspc)

        except IndexError:
            print("Problems finding adjoint source: "+adjtf)

    return(adjt_spect)


def compute_kernel(wf1str,wf2str,adjt,
    src,source_conf,insta,params,basis):

    
    
    kern = np.zeros((params.filtcnt,params.n_traces,
        source_conf['spectra_nr_parameters']))

    ########################################################################
    # Get the adjoint spectra from files
    ######################################################################## 
    adjt_spect = get_adjoint_spectra(adjt,params)

    if (adjt_spect==0.0).sum() == params.filtcnt * params.n_freq:
        print('No adjoint source found for: '+os.path.basename(wf1str)
            +' -- '+os.path.basename(wf2str))
        return()
    

    ########################################################################
    # Compute the kernels
    ######################################################################## 

    if insta:
        src_grid = np.load(os.path.join(source_conf['project_path'],
            'sourcegrid.npy'))
        # open database
        dbpath = json.load(open(os.path.join(source_conf['project_path'],
            'config.json')))['wavefield_path']
        # open and determine Fs, nt
        db = instaseis.open_db(dbpath)
        # get receiver locations
        lat1 = geograph_to_geocent(float(wf1[2]))
        lon1 = float(wf1[3])
        rec1 = instaseis.Receiver(latitude=lat1,longitude=lon1)
        lat2 = geograph_to_geocent(float(wf2[2]))
        lon2 = float(wf2[3])
        rec2 = instaseis.Receiver(latitude=lat2,longitude=lon2)

    else:
        wf1 = WaveField(wf1str)
        wf2 = WaveField(wf2str)
        
    ########################################################################
    # Loop over locations
    ######################################################################## 
    
    for i in range(params.n_traces):

        ####################################################################
        # Get synthetics
        #################################################################### 
        if insta:
        # get source locations
            lat_src = geograph_to_geocent(src_grid[1,i])
            lon_src = src_grid[0,i]
            fsrc = instaseis.ForceSource(latitude=lat_src,
                longitude=lon_src,f_r=1.e12)
            
            s1 = np.ascontiguousarray(db.get_seismograms(source=fsrc,
                receiver=rec1,
                dt=1./source_conf['sampling_rate'])[0].data*taper)
            s2 = np.ascontiguousarray(db.get_seismograms(source=fsrc,
                receiver=rec2,
                dt=1./source_conf['sampling_rate'])[0].data*taper)
        
        else:
            s1 = np.ascontiguousarray(wf1.data[i,:]*params.taper)
            s2 = np.ascontiguousarray(wf2.data[i,:]*params.taper)

        spec1 = np.fft.rfft(s1,params.n)
        spec2 = np.fft.rfft(s2,params.n)
        corr_temp = np.multiply(np.conjugate(spec1),spec2)
        
    #######################################################################
    # Apply the 'adjoint spectra'
    #######################################################################
        for ix_f in range(params.filtcnt):

            
                # instead of dot product: project to basis
                
            kern[ix_f,i,:] = basis.coeff(corr_temp*
                    adjt_spect[ix_f,:] * params.delta) 
                
    if not insta:
        wf1.file.close()
        wf2.file.close()

    return(kern)

       

def correlation_pairs_to_compute(source_config):

    auto_corr = False # default value
    try:
        auto_corr = source_config['get_auto_corr']
    except KeyError:
        pass

    p = define_correlationpairs(source_config['project_path'],auto_corr)
    print('Nr all possible kernels %g ' %len(p))
    
    # Remove pairs for which no observation is available
    if source_config['model_observed_only']:
        directory = os.path.join(source_config['source_path'],
            'observed_correlations')
        p = rem_no_obs(p,source_config,directory=directory)
        print('Nr kernels after checking actual observations %g ' %len(p))

    return(p)

def run_kern(source_configfile,step,ignore_network=False):

    t0 = time.time()

    #######################################################################
    # MPI setup
    #######################################################################
    # NEW: rank 0 collects kernels and computes gradient.
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    #######################################################################
    # Configuration
    #######################################################################
    source_config=json.load(open(source_configfile))
    measr_config = json.load(open(os.path.join(source_config['source_path'],
        'measr_config.json')))
    config = json.load(open(os.path.join(source_config['project_path'],
        'config.json')))

    insta = config['instaseis']
    bandpass = measr_config['bandpass']
    print("Rank %g got config: %.4f sec" %(rank,time.time()-t0))

    #######################################################################
    # Parameters
    #######################################################################

    params = kernel_params()

    if rank == 0:
        output_path = os.path.join(source_config['source_path'],
            'step_'+str(step),'grad','grad_all.npy')

        p = correlation_pairs_to_compute(source_config)

        n_time, n, n_corr, Fs = get_ns(config,source_config,insta)

        src_grid = np.load(os.path.join(source_config['project_path'],
            'sourcegrid.npy'))
        n_traces = src_grid.shape[-1]

        filtcnt = get_nr_measurements(bandpass)
        kern = np.empty((filtcnt,n_traces,
            source_config['spectra_nr_parameters']))
        grad = np.zeros((filtcnt,n_traces,
            source_config['spectra_nr_parameters']))
    else:

        n_time = 0
        n = 0
        n_corr = 0
        Fs = 0.0
        n_traces = 0
        p = None


    p = comm.bcast(p,root=0)
    params.p = p


    n_traces = comm.bcast(n_traces,root=0)
    params.n_traces = n_traces

    n_time = comm.bcast(n_time,root=0)
    n = comm.bcast(n,root=0)
    params.n = n
    n_corr = comm.bcast(n_corr,root=0)
    params.n_corr = n_corr
    Fs = comm.bcast(Fs,root=0)
    params.Fs = Fs
    params.delta = 1./params.n
    
    if params.n % 2 == 1: #uneven
        params.n_freq = int((params.n+1)/2)
    else:
        params.n_freq = int(params.n/2+1)

    params.step = int(step)
    params.filtcnt = get_nr_measurements(bandpass)
    
    # use a one-sided taper: The seismogram probably has a non-zero end, 
    # being cut off whereever the solver stopped running.
    params.taper = cosine_taper(n_time,p=0.01)
    params.taper[0:n_time//2] = 1.0
    

    print("Rank %g set up parameters: %.4f sec" %(rank,time.time()-t0))

    
    
    #######################################################################
    # Correlation pair loop
    #######################################################################
    
    if rank > 0:

        basis = BasisFunction(basis_type=source_config['spectra_decomposition'],
        K=source_config['spectra_nr_parameters'],N=params.n_freq)

        num_pairs = int( ceil(float(len(p))/float(size-1)) )
        p_p = p[ (rank-1)*num_pairs : rank*num_pairs]
        print("Rank %g working on pair nr. %g to %g of %g." 
        %(rank,(rank-1)*num_pairs,
        rank*num_pairs,len(p)))

        # loop proper
        for cp in p_p:
            
            try:
                wf1,wf2,src,adjt = paths_input(cp,source_config,
                    params.step,ignore_network,insta)
                
            except:
                print('Could not find necessary input. \
\nCheck if files {},{} and step_{}/base_model.h5 file are available.'.format(
wf1,wf2,step))
                continue


            kern = compute_kernel(wf1,wf2,adjt,src,source_config,
                insta,params,basis)

            print("Rank %g computed kernel %s,%s: %.4f sec" %(rank,
                os.path.basename(wf1),os.path.basename(wf2),time.time()-t0))

            # pass back to rank 0
            comm.Send(kern, dest=0, tag=rank)

    else:

        # rank 0 waits around for kernels to be passed back to it.
        status = MPI.Status()
        print(time.time()-t0)
        comm.Recv(kern,source=MPI.ANY_SOURCE,tag=MPI.ANY_TAG,status=status)
        print(time.time()-t0)
        grad += kern

    comm.Barrier()

    # Finally: Save the gradient.
    if rank == 0:
        np.save(output_path,grad)
