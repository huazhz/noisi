import numpy as np
from math import sqrt, pi, ceil
import sys
from mpi4py import MPI
from warnings import warn
import json
try:
    from noisi.util.plot import plot_grid
except:
    pass
import os
# Try yet another: sort of Gaussian convolution, but determining the distance
# in cartesian coordinates.



def get_distance(gridx,gridy,gridz,x,y,z):
    #def distance_function(x1,y1,z1,x2,y2,z2):
    #    return sqrt((x2-x1)**2+(y2-y1)**2+(z2-z1)**2)
    #dist = np.array([distance_function()])
    xd = gridx - x
    yd = gridy - y
    zd = gridz - z
    
    return np.sqrt(np.power(xd,2)+np.power(yd,2)+np.power(zd,2))

def spherical_to_cartesian(coords):

    r=6371000.
    theta = np.deg2rad(-coords[1] + 90.) 
    phi = np.deg2rad(coords[0] + 180.)

    x = r*np.sin(theta) * np.cos(phi)
    y = r*np.sin(theta) * np.sin(phi)
    z = r*np.cos(theta)
    return(x,y,z)



def smooth_gaussian(inputfile,outputfile,coordfile,sigma,cap,thresh):

    #
    r = 6371000.
    # initialize parallel comm
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    # load input
    coords = np.load(coordfile)
    input_array = np.load(inputfile)

    # initialize output array
    smoothed_array = np.zeros(input_array.shape)
    smoothed_array_out = np.zeros(input_array.shape)
    
    # transform the coordinates
    x,y,z = spherical_to_cartesian(coords)
    if rank == 0:
        print('Attention, smoothing currently pretends the Earth is round.')

    # chunk size
    chunk_size = ceil(len(coords[0])/size)


    for j in range(input_array.shape[0]):
        print(80*'*')
        print(j)
        for i in range(input_array.shape[-1]):
            print(i)
            array_in = input_array[j,:,i]
            
            # apply the cap
            perc_up = np.percentile(array_in,cap,overwrite_input=False)
            perc_dw = np.percentile(array_in,100-cap,overwrite_input=False)
            values = np.clip(array_in,perc_dw,perc_up)

            
            try:
                sig = sigma[i]
            except IndexError:
                if rank == 0: 
                    print("Less values for smoothing length given than K,\
                        using last value for remaining k.")
                sig = sigma[-1]
                
            # prepare Gaussian
            a = 1./(sig*sqrt(2.*pi))
            b = 2.*sig**2
            # rank x works on this part:
            for k in range(rank*chunk_size,(rank+1)*chunk_size):
                if k%1000==0:
                    print(k)    
                try:
                    # Gaussian smoothing
                    dist = get_distance(x,y,z,x[k],y[k],z[k])
                    
                except IndexError:
                    break

                weight = a * np.exp(-(dist**2)/b)
                idx = weight >= (thresh * weight.max())
                smoothed_array[j,k,i] = np.sum(np.multiply(weight[idx],
                    values[idx])) / idx.sum()

    comm.Barrier()
    comm.Reduce(smoothed_array,smoothed_array_out,MPI.SUM,root=0)

    if rank == 0:
        np.save(outputfile,smoothed_array_out)



if __name__=='__main__':

    # pass in: input_file, output_file, coord_file, sigma
    # open the files
    source_dir = sys.argv[1]
    step = sys.argv[2]

    source_config = json.load(open(os.path.join(source_dir,
        'source_config.json')))
    
    inputfile = os.path.join(source_dir,'step_'+str(step),'grad','grad_all.npy')
    outputfile = os.path.join(source_dir,'step_'+str(step),'grad',
        'grad_smooth.npy')
    coordfile = os.path.join(source_config['project_path'],'sourcegrid.npy')
    sigma = source_config['smoothing_lengths_m']
    sigma = [float(s) for s in sigma]
    cap_percentile = source_config['smoothing_cap_percentile']

    # distances contributing less than this threshold are neglected 
    thresh = source_config['smoothing_threshold_fraction_of_max']
    
    smooth_gaussian(inputfile,outputfile,coordfile,sigma,cap_percentile,thresh)


    
