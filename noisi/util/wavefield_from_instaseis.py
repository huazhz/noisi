# create a wavefield from instaseis
from mpi4py import MPI
import instaseis
import h5py
import os
import sys
from pandas import read_csv
import numpy as np
import json
from noisi.util.geo import geograph_to_geocent


comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()


# get config
source_config=json.load(open('source_config.json'))
config = json.load(open('../config.json'))
Fs = source_config['sampling_rate']
path_to_db = config['wavefield_path']
channel = source_config['channel']

# read sourcegrid
f_sources = np.load('../sourcegrid.npy')
ntraces = f_sources.shape[-1]

# open the database
db = instaseis.open_db(path_to_db)

# get: synthetics duration and sampling rate in Hz
stest = db.get_seismograms(source=instaseis.ForceSource(latitude=0.0,
            longitude=0.0),receiver=instaseis.Receiver(latitude=10.,
            longitude=0.0),dt=1./source_config['sampling_rate'])[0]
ntimesteps = stest.stats.npts


# read station from file
stationlist = read_csv('../stationlist.csv')
net = stationlist.at[rank,'net']
sta = stationlist.at[rank,'sta']
lat = stationlist.at[rank,'lat']
lon = stationlist.at[rank,'lon']
print(net,sta,lat,lon)
# output directory:
if rank == 0:
	os.system('mkdir -p wavefield_processed')

f_out_name = '{}.{}..MXZ.h5'.format(net,sta,channel)
f_out_name = os.path.join('wavefield_processed',f_out_name)

f_out = h5py.File(f_out_name, "w")


# DATASET NR 1: STATS
stats = f_out.create_dataset('stats',data=(0,))
stats.attrs['reference_station'] = '{}.{}'.format(net,sta)
stats.attrs['data_quantity'] = config['synt_data']
stats.attrs['ntraces'] = ntraces
stats.attrs['Fs'] = Fs
stats.attrs['nt'] = int(ntimesteps)

# DATASET NR 2: Source grid
sources = f_out.create_dataset('sourcegrid',data=f_sources[0:2])


lat1 = geograph_to_geocent(float(lat))
lon1 = float(lon)
rec1 = instaseis.Receiver(latitude=lat1,longitude=lon1)

# DATASET Nr 3: Seismograms itself
traces = f_out.create_dataset('data',(ntraces,ntimesteps),dtype=np.float32)

if channel[-1] == 'Z':
	c_index = 0
elif channel[-1] == 'R':
	c_index = 1
elif channel[-1] == 'T':
	c_index = 2



# jump to the beginning of the trace in the binary file
for i in range(ntraces):
    if i%1000 == 1:
        print('Converted %g of %g traces' %(i,ntraces))
    # read station name, copy to output file
   
    lat_src = geograph_to_geocent(f_sources[1,i])
    lon_src = f_sources[0,i]
    fsrc = instaseis.ForceSource(latitude=lat_src,
                    longitude=lon_src,f_r=1.e12)

    values =  db.get_seismograms(source=fsrc,receiver=rec1,dt=1./Fs)
    if c_index in [1,2]:
    	baz = gps2dist_azimuth(lat_src,lon_src,lat,lon)
    	values.rotate('NE->RT',back_azimuth=baz)

    values = values[c_index]
    values.taper(0.01)

    if config['synt_data'] in ['VEL','ACC']:
    	values.differentiate()
    	if config['synt_data'] == 'ACC':
    		values.differentiate()
    # Save in traces array
    traces[i,:] = values.data
    
    
f_out.close()

