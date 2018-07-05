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
from obspy.geodetics import gps2dist_azimuth

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()


# get config
source_config=json.load(open('source_config.json'))
config = json.load(open('../config.json'))
Fs = source_config['sampling_rate']
path_to_db = config['wavefield_path']

chas = config['station_channels']
channels = []
for c in chas:
    channels.append(c[-1].upper())


source_component = 'MXZ'

seismo = config['synt_data']
if seismo == 'DIS':
    seismo_kind = 'displacement'
elif seismo == 'VEL':
    seismo_kind = 'velocity'
elif seismo == 'ACC':
    seismo_kind = 'acceleration'


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

if len(channels) == 3:
    cha = 'ALL'
elif len(channels) == 1:
    cha = chas[0]
else:
    cha = channels[0]+'_'+channels[1]
f_out_name = '{}.{}...h5'.format(net,sta,cha)
f_out_name = os.path.join('wavefield_processed',f_out_name)


if not os.path.exists(f_out_name):

    startindex = 0

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
    if 'Z' in channels:
        traces_z = f_out.create_dataset('data_z',(ntraces,ntimesteps),
        dtype=np.float32)
    if 'N' in channels:
        traces_n = f_out.create_dataset('data_n',(ntraces,ntimesteps),
        dtype=np.float32)
    if 'E' in channels:
        traces_e = f_out.create_dataset('data_e',(ntraces,ntimesteps),
        dtype=np.float32)
    # elif if channel.upper() == 'ALL':
    #     traces_z = f_out.create_dataset('data_z',(ntraces,ntimesteps),
    #     dtype=np.float32)
    #     traces_n = f_out.create_dataset('data_n',(ntraces,ntimesteps),
    #     dtype=np.float32)
    #     traces_e = f_out.create_dataset('data_e',(ntraces,ntimesteps),
    #     dtype=np.float32)


else:

    f_out = h5py.File(f_out_name, "r+")
    if 'Z' in channels:
        startindex = len(f_out['data_z']) 
    elif 'E' in channels:
        startindex = len(f_out['data_e']) 
    elif 'N' in channels:
        startindex = len(f_out['data_n']) 


# jump to the beginning of the trace in the binary file
for i in range(startindex,ntraces):
    if i%(0.1*ntraces) == 1:
        print('Converted %g of %g traces' %(i,ntraces))
    # read station name, copy to output file
   
    lat_src = geograph_to_geocent(f_sources[1,i])
    lon_src = f_sources[0,i]

    ######### ToDo! Right now, only either horizontal or vertical component sources ##########
    if source_component[-1] in ['E','N']:
        fsrc = instaseis.ForceSource(latitude=lat_src,
                    longitude=lon_src,f_t=1.e09,f_p=1.e09)
    elif source_component[-1] == 'Z':
        fsrc = instaseis.ForceSource(latitude=lat_src,
                    longitude=lon_src,f_r=1.e09)

    values =  db.get_seismograms(source=fsrc,receiver=rec1,dt=1./Fs,
        kind=seismo_kind)
    
    # I don't think rotation is practical (cross-correlations require 
    # different orientation anyway). Todo: Figure out horizontal comp. source
    #if c_index in [1,2]:
   # 	baz = gps2dist_azimuth(lat_src,lon_src,lat,lon)[2]
   # 	values.rotate('NE->RT',back_azimuth=baz)


    # Save in traces array
    if 'Z' in channels:
        traces_z[i,:] = values[0].data
    if 'E' in channels:
        traces_e[i,:] = values[2].data
    if 'N' in channels:
        traces_n[i,:] = values[1].data
    # elif channel.upper() == 'ALL':
    #     traces_z[i,:] = values[0].data
    #     traces_n[i,:] = values[1].data
    #     traces_e[i,:] = values[2].data

    
    
f_out.close()

