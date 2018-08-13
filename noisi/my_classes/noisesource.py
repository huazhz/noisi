import numpy as np
import h5py

from scipy.signal import hann, tukey
import os
try:
    from noisi.util.plot import plot_grid
except ImportError:
    print('Plotting unavailable, is basemap installed?')
from noisi.my_classes.basisfunction import BasisFunction
from noisi.util.geo import get_spherical_surface_elements

class NoiseSource(object):
    """
   'model' of the noise source that comes in terms of a couple of basis 
    functions and associated weights. The NoiseSource object should contain a 
    function to determine weights from a (kernel? source model?), and to expand from weights and basis 
    functions.
    
    """
    def __init__(self,model,w='r+'):
            
        # Model is an hdf5 file which contains the basis and weights of the source model!
        
       
        try:
            self.model = h5py.File(model,w)
            print(self.model)
            
        except IOError:
            msg = 'Unable to open model file '+model
            raise IOError(msg)

        self.src_loc = self.model['coordinates']
        self.freq = self.model['frequencies']
        self.N = int(len(self.freq))
        
        try:
            self.f_min = round(self.model['model'].attrs['f_min'],6)
            self.f_max = round(self.model['model'].attrs['f_max'],6)
        except KeyError:
            self.f_min = self.freq[0]
            self.f_max = self.freq[-1]

        try:
            self.basis_type = self.model['model'].attrs['spectral_basis']
            K = self.model['model'][:].shape[1]
            
            self.spectral_basis = BasisFunction.initialize(
                self.basis_type,K,N=self.N,freq=self.freq,
                fmin=self.f_min,fmax=self.f_max)
            self.spectral_coefficients = self.model['model'] # don't read into memory
            # frequency range of interest

            
        except KeyError:
           # Presumably, these arrays are small and will be used very often 
           # --> good to have in memory
           self.basis_type = 'discrete'
           self.spectral_coefficients = self.model['distr_basis'][:] #distr_basis
           self.spectral_basis = self.model['spect_basis'][:]
           self.N = self.spectral_basis.shape[-1]
            # self.distr_weights = self.model['distr_weights'][:]
            # distr basis currently unused

        # The surface area of each grid element...new since June 18
        try:
            self.surf_area = self.model['surf_areas'][:]
        except KeyError:
            self.surf_area = np.ones(self.src_loc.shape[-1])
            


        
    def __enter__(self):
        return self
    
    def __exit__(self,type,value,traceback):
        
        if self.model is not None:
            self.model.close()
            #ToDo: Check what parameters/data should be written before file closed

    def filter(self,freq_min,freq_max,window_type='tukey'):

        window = np.zeros(self.freq.shape)
        ix_1 = np.argmin(np.abs(self.freq[:]-freq_min))
        ix_2 = np.argmin(np.abs(self.freq[:]-freq_max))
        
        if window_type == 'tukey':
            window[ix_1:ix_2] = tukey(ix_2-ix_1)
        elif window_type == 'hann':
            window[ix_1:ix_2] = hann(ix_2-ix_1)
        else:
            raise NotImplementedError("Unknown window type.")

        filt_output = np.zeros(self.src_loc.shape[-1])

        for i in range(self.src_loc.shape[-1]):

            filt_output[i] = np.dot(self.spectral_basis.expand(
                self.spectral_coefficients[i,:]),window)/self.freq.shape[0]

        return(filt_output)





    def get_spect(self,iloc,N=None):
        # return one spectrum in location with index iloc
        # The reason this function is for one spectrum only is that the 
        # entire gridded matrix of spectra by location is most 
        # probably pretty big.
        
        if N == None:
            N = self.N
        if N == None:
            raise ValueError("You must set N (number of samples.)")

        try:
            spectrum = self.spectral_basis.expand(self.
                spectral_coefficients[iloc,:],N)
            spectrum = np.clip(spectrum,0.0,spectrum.max())
        except AttributeError:
            spectrum = np.dot(self.spectral_coefficients[:,iloc],
                self.spectral_basis)

        return(spectrum)
    
    
    def plot(self,**options):
        
        # plot the distribution
       
        if self.spectral_coefficients.shape[0] < 5:       
            for m in range(self.spectral_coefficients.shape[0]): 
                filename = 'noise_source_coeffs_{}.png'.format(m)
                plot_grid(self.src_loc[0],self.src_loc[1],
                    self.spectral_coefficients[m,:],outfile=filename,
                    **options)

        else:
            max_freq = np.zeros(self.src_loc.shape[-1])
            for i in range(self.src_loc.shape[-1]):

                spectrum = self.get_spect(i)
                max_freq[i] = self.freq[spectrum.argmax()]

            plot_grid(self.src_loc[0],self.src_loc[1],
                max_freq,outfile='noise_source_dominant_freq.png',
                sequential=True,**options)

    
    # def get_spectrum(self,iloc):
    #    # Return the source spectrum at location nr. iloc
       
    #    #if self.file is not None:
    #    #    return self.sourcedistr[iloc] * self.spectra[iloc,:]
    #    if self.model is not None:
    #        return self.spectr.
    #        # (Expand from basis fct. in model)
    # def get_sourcedistr(self,i):
    #    # Return the spatial distribution of max. PSD
    #    if self.file is not None:
    #        return np.multiply(self.sourcedistr[:],self.spectra[:,i])
    #    else:
    #        raise NotImplementedError
   
