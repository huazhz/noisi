from __future__ import print_function
import numpy as np
import os
import h5py
#from obspy import Trace
try:
    from noisi.util import plot
except ImportError:
    print('Plotting unavailable, is basemap installed?')
from noisi.util import filter
try:
    from scipy.signal import sosfilt
except ImportError:
    from obspy.signal._sosfilt import _sosfilt as sosfilt
try:
    from scipy.fftpack import next_fast_len
except ImportError:
    from noisi.borrowed_functions.scipy_next_fast_len import next_fast_len
#from scipy.signal.signaltools import _next_regular
from obspy.signal.invsim import cosine_taper
from obspy.signal.filter import integer_decimation
import click
from warnings import warn

# dictionary to map cross-correlation components to their output names
cc_comp_dict = {'datadata':('Z','Z'),
                'data_zdata_z':('Z','Z'),
                'data_zdata_e':('Z','T'),
                'data_zdata_r':('Z','R'),
                'data_edata_z':('T','Z'),
                'data_edata_e':('T','T'),
                'data_edata_r':('T','R'),
                'data_ndata_z':('R','Z'),
                'data_ndata_e':('R','T'),
                'data_ndata_n':('R','R'),}

class WaveField(object):
    """
    Object to handle database of stored wavefields.
    Basically, just a few methods to work on wavefields stored in an hdf5 file. 
    The stored seismograms have to have sampling rate as attribute Fs and number of time steps as attribute ntime; They have to have an ID of format net.sta.loc.cha    
    """
    def __init__(self,file,sourcegrid=None,w='r'):

        self.w = w
        
        try:   
            self.file = h5py.File(file, self.w)
        except IOError:
            msg = 'Unable to open input file ' + file
            raise IOError(msg)

        self.components = {}
        #self.component_names = []
        
        self.stats = dict(self.file['stats'].attrs)
        self.sourcegrid = self.file['sourcegrid']
        
        if "data" in list(self.file.keys()):
            self.data = self.file['data']
            self.components['data'] = self.data
            #self.component_names += 'data'

        elif "data_z" in list(self.file.keys()):
            self.data_z = self.file['data_z']
            self.components['data_z'] = self.data_z
            #self.component_names += 'data_z'
            
        if "data_e" in list(self.file.keys()):
            self.data_e = self.file['data_e']
            self.components[data_e] = self.data_e
            #self.component_names += 'data_e'

        if "data_n" in list(self.file.keys()):
            self.data_n = self.file['data_n']
            self.components['data_n'] = self.data_n
            #self.component_names += 'data_n'
               
        print(self.file)
        
        #ToDo handle complex
   # Thought about using a class method here, but need a copy of the stats!
    def copy_setup(self,newfile,nt=None,ntraces=None,w='r+'):
        

        # Copy the stats and sourcegrid to a new file with empty (all-zero) arrays for seismograms
        
        # Shape of the new array:
        shape = list(np.shape(self.data))
        if ntraces is not None:
            shape[0] = ntraces
        if nt is not None:
            shape[1] = nt
        shape = tuple(shape)
        
        # Create new file
        file = h5py.File(newfile, 'w-')
       
        # Copy metadata
        stats = file.create_dataset('stats',data=(0,))
        for (key,value) in self.stats.items():
            file['stats'].attrs[key] = value
        
        # Ensure that nt is kept as requested
        if nt is not None and nt != self.stats['nt']:
            file['stats'].attrs['nt'] = nt

        #stats.attrs['reference_station'] = self.stats['refstation']
        #stats.attrs['data_quantity'] = self.stats['data_quantity']
        #stats.attrs['ntraces'] = shape[0]
        #stats.attrs['Fs'] = self.stats['Fs']
        #stats.attrs['nt'] = shape[1]
        
        file.create_dataset('sourcegrid',data=self.sourcegrid[:].copy()) 
        
        # Initialize data arrays
        #if complex:
        #    file.create_dataset('real',shape,dtype=np.float32)
        #    file.create_dataset('imag',shape,dtype=np.float32)   
        #else:
        for comp_name,comp in self.components.items():
            file.create_dataset(comp_name,shape,dtype=np.float32)
        # if "data" in list(self.file.keys()):
        #     file.create_dataset('data',shape,dtype=np.float32)
        # if "data_z" in list(self.file.keys()):
        #     file.create_dataset('data_z',shape,dtype=np.float32)
            
        # if "data_e" in list(self.file.keys()):
        #     file.create_dataset('data_e',shape,dtype=np.float32)
            
        # if "data_n" in list(self.file.keys()):
        #     file.create_dataset('data_n',shape,dtype=np.float32)
                
        
        print('Copied setup of '+self.file.filename)
        file.close()
         
        return(WaveField(newfile,w=w))
    
    #def copy_setup_real_to_complex(self,newfile,w='r+'):
    #    #Copy the stats and sourcegrid to a new file with empty (all-zero) arrays for seismograms
    #    #extend seismograms to spectra to fit the expected length of zero-padded FFT, and add real as well as imag. part
    #    file = h5py.File(newfile, 'w')
    #    file.create_dataset('stats',data=(0,))
    #    for (key,value) in self.stats.items():
    #        file['stats'].attrs[key] = value
    #    nfft = _next_regular(2*self.stats['nt']-1)
    #    shape = (self.stats['ntraces'],nfft//2+1)
    #    file.create_dataset('sourcegrid',data=self.sourcegrid[:].copy())    
    #    file.create_dataset('real',shape,dtype=np.float32)
    #    file.create_dataset('imag',shape,dtype=np.float32)
    #    
    #    file.close()
    #    return WaveField(newfile,complex=True,w=w)
    #
    
    def truncate(self,newfile,truncate_after_seconds):
        
        nt_new = int(round(truncate_after_seconds * self.stats['Fs']))
    
        with self.copy_setup(newfile,nt=nt_new) as wf:
        
            for i in range(self.stats['ntraces']):
          
                for comp_name,comp in wf.components.items():
                    comp[i,:] = self.components[comp_name][i,0:nt_new].copy()
                
    def get_correlation_components(self):

        # get the possible components of cross-correlation
        # given the input channels

        corr_comps = []
        corr_comp_names = []
        

        for c in list(self.components.keys()):
            for k in list(self.components.keys()):
                corr_comps.append([self.components[c],self.components[k]])
                corr_comp_names.append([c,k])

        # the horizontal components will be rotated for each station pair
        # so that from the correlation of E and E component, 
        # we obtain a TT correlation etc.
        corr_comp_output_names = [cc_comp_dict[ccn[0]+ccn[1]] 
        for ccn in corr_comp_names]

        return corr_comps, corr_comp_names, corr_comp_output_names


    def filter_all(self,type,overwrite=False,zerophase=True,outfile=None,**kwargs):
        
        if type == 'bandpass':
            sos = filter.bandpass(df=self.stats['Fs'],**kwargs)
        elif type == 'lowpass':
            sos = filter.lowpass(df=self.stats['Fs'],**kwargs)
        elif type == 'highpass':
            sos = filter.highpass(df=self.stats['Fs'],**kwargs)
        else:
            msg = 'Filter %s is not implemented, implemented filters: bandpass, highpass,lowpass' %type
            raise ValueError(msg)
        
        if not overwrite:
            # Create a new hdf5 file of the same shape
            newfile = self.copy_setup(newfile=outfile)
        else:
            # Call self.file newfile
            newfile = self#.file
        
        for i in range(self.stats['ntraces']):
                # Filter each trace

            for comp_name,comp in newfile.components.items():
                if zerophase:
                    firstpass = sosfilt(sos, self.components[comp_name][i,:]) # Read in any case from self.data
                    comp[i,:] = sosfilt(sos,firstpass[::-1])[::-1] # then assign to newfile, which might be self.file
                else:
                    comp[i,:] = sosfilt(sos,
                        self.components[comp_name][i,:])
                
            
                
        if not overwrite:
           print('Processed traces written to file %s, file closed, \
                  reopen to read / modify.' %newfile.file.filename)
           
           newfile.file.close()
            

    def decimate(self,decimation_factor,outfile,taper_width=0.005):
        """
        Decimate the wavefield and save to a new file 
        """
        
        fs_old = self.stats['Fs']
        freq = self.stats['Fs'] * 0.4 / float(decimation_factor)

        # Get filter coeff
        sos = filter.cheby2_lowpass(fs_old,freq)

        # figure out new length
        temp_trace = integer_decimation(self.data[0,:], decimation_factor)
        n = len(temp_trace)
       

        # Get taper
        # The default taper is very narrow, because it is expected that the traces are very long.
        taper = cosine_taper(self.stats['nt'],p=taper_width)

       
        # Need a new file, because the length changes.
        with self.copy_setup(newfile=outfile,nt=n) as newfile:

            for i in range(self.stats['ntraces']):
                
                for comp_name,comp in newfile.components.items():
                    temp_trace = sosfilt(sos,taper*
                        self.components[comp_name][i,:])
                    comp[i,:] = integer_decimation(temp_trace, decimation_factor)
                
        
            newfile.stats['Fs'] = fs_old / float(decimation_factor)



    # def space_integral(self,weights=None):
    #     # ToDo: have this checked; including spatial sampling!
    #     # ToDo: Figure out how to assign the metadata...buh
    #     trace = Trace()
    #     trace.stats.sampling_rate = self.stats['Fs']
        
    #     # ToDo: Thinking about weights
    #     if not self.complex:
    #         if weights: 
    #             trace.data = np.trapz(np.multiply(self.data[:],weights[:]),axis=0)
    #         else:
    #             trace.data = np.trapz(self.data[:],axis=0)
    #     #oDo complex wavefield
    #     else:
    #         if weights: 
    #             trace.data_i = np.trapz(np.multiply(self.data_i[:],weights[:]),axis=0)
    #             trace.data_r = np.trapz(np.multiply(self.data_r[:],weights[:]),axis=0)
    #         else:
    #             trace.data_i = np.trapz(self.data_i[:],axis=0)
    #             trace.data_r = np.trapz(self.data_r[:],axis=0)
            
    #     return trace
            
    
    def get_snapshot(self,t,resolution=1,component='z'):
        
        #ToDo: Ask someone who knows h5py well how to do this in a nice way!
        
        t_sample = int(round(self.stats['Fs'] * t))

        # find component
        try:
            c = self.components['data_'+component]
        except KeyError:
            try:
                c = self.components['data']
            except KeyError:
                print('Nothing found to plot')
                return()

        if t_sample >= np.shape(c)[1]:
            warn('Requested sample is out of bounds, resetting to last sample.')
            t_sample = np.shape(c)[1]-1
        if resolution == 1:
            snapshot = c[:,t_sample]
        else:
            snapshot = c[0::resolution,t_sample] #0:len(self.data[:,0]):resolution
       

        # if component == "z" and "data" in list(self.file.keys()):
        #     if t_sample >= np.shape(self.data)[1]:
        #         warn('Requested sample is out of bounds, resetting to last sample.')
        #         t_sample = np.shape(self.data)[1]-1
        #     if resolution == 1:
        #         snapshot = self.data[:,t_sample]
        #     else:
        #         snapshot = self.data[0::resolution,t_sample] #0:len(self.data[:,0]):resolution
        # elif component == "z" and "data_z" in list(self.file.keys()):
        #     if t_sample >= np.shape(self.data_z)[1]:
        #         warn('Requested sample is out of bounds, resetting to last sample.')
        #         t_sample = np.shape(self.data_z)[1]-1
        #     if resolution == 1:
        #         snapshot = self.data_z[:,t_sample]
        #     else:
        #         snapshot = self.data_z[0::resolution,t_sample] #0:len(self.data[:,0]):resolution
        # if component == 'e' and "data_e" in list(self.file.keys()):
        #     if t_sample >= np.shape(self.data_e)[1]:
        #         warn('Requested sample is out of bounds, resetting to last sample.')
        #         t_sample = np.shape(self.data_e)[1]-1
        #     if resolution == 1:
        #         snapshot = self.data_e[:,t_sample]
        #     else:
        #         snapshot = self.data_e[0::resolution,t_sample] #0:len(self.data[:,0]):resolution
        # if component == 'n' and "data_n" in list(self.file.keys()):
        #     if resolution == 1:
        #         snapshot = self.data_n[:,t_sample]
        #     else:
        #         snapshot = self.data_n[0::resolution,t_sample] #0:len(self.data[:,0]):resolution
        

        print('Got snapshot')
        
        return snapshot
    
    #ToDo put somewhere else    
    def plot_snapshot(self,t,resolution=1,component='z',**kwargs):
        
        if self.sourcegrid is None:
            msg = 'Must have a source grid to plot a snapshot.'
            raise ValueError(msg)
        
        # ToDo: Replace all the hardcoded geographical boundary values!
        map_x = self.sourcegrid[0][0::resolution]
        map_y = self.sourcegrid[1][0::resolution]
                                 
        plot.plot_grid(map_x,map_y,self.get_snapshot(t,resolution=resolution,component=component),**kwargs)
    
    def update_stats(self):
        
        if self.w != 'r':
            print('Updating stats...')
            self.file['stats'].attrs['ntraces'] = len(self.data[:,0])
            self.file['stats'].attrs['nt'] = len(self.data[0,:])

            if 'stats' not in self.file.keys():
                self.file.create_dataset('stats',data=(0,))
            for (key,value) in self.stats.items():
                self.file['stats'].attrs[key] = value
            
            #print(self.file['stats'])
            #self.file.flush()
    
    #def write_sourcegrid(self):
     #   self.file.create_dataset('sourcegrid',data=self.sourcegrid)
     #   self.file.flush()
    

    def __enter__(self):
        return self
    
    def __exit__(self,type,value,traceback):
       
        self.update_stats()
        
        #ToDo update the stats
        
        self.file.close()
        
        
        
    
