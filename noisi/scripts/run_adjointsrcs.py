import os
import numpy as np
from math import log, pi
import click
import json
from scipy.signal import hilbert
from glob import glob
from obspy import read, Trace
from obspy.geodetics import gps2dist_azimuth

from noisi.scripts import adjnt_functs as af
from noisi.util.windows import get_window, my_centered, snratio


def get_station_info(stats):

    sta1 = '{}.{}.{}.{}'.format(stats.network,stats.station,stats.location,
    stats.channel)
    sta2 = '{}.{}.{}.{}'.format(stats.sac.kuser0.strip(),stats.sac.kevnm.strip(),
    stats.sac.kuser1.strip(),stats.sac.kuser2.strip())
    lat1 = stats.sac.stla
    lon1 = stats.sac.stlo
    lat2 = stats.sac.evla
    lon2 = stats.sac.evlo
    dist = stats.sac.dist
    az = gps2dist_azimuth(lat1,lon1,lat2,lon2)[2]
    
    
    return([sta1,sta2,lat1,lon1,lat2,lon2,dist,az])

def get_synthetics_filename(obs_filename,synth_location='',fileformat='sac',synth_channel_basename='MX'):

    inf = obs_filename.split('--')

    if len(inf) == 1:
        # old station name format
        inf = obs_filename.split('.')
        net1 = inf[0]
        sta1 = inf[1]
        cha1 = inf[3]
        net2 = inf[4]
        sta2 = inf[5]
        cha2 = inf[7]
    elif len(inf) == 2:
        # new station name format
        inf1 = inf[0].split('.')
        inf2 = inf[1].split('.')
        net1 = inf1[0]
        sta1 = inf1[1]
        net2 = inf2[0]
        sta2 = inf2[1]
        cha1 = inf1[3]
        cha2 = inf2[3]


    cha1 = synth_channel_basename + cha1[-1]
    cha2 = synth_channel_basename + cha2[-1]

    synth_filename = '{}.{}.{}.{}--{}.{}.{}.{}.{}'.format(net1,sta1,synth_location,
        cha1,net2,sta2,synth_location,cha2,fileformat)
    print(synth_filename)
    return(synth_filename)


def adjointsrcs(source_config,mtype,step,**options):
    
    """
    Get 'adjoint source' from noise correlation data and synthetics. 
    options: g_speed,window_params (only needed if mtype is ln_energy_ratio or enery_diff)
    """
    
    
    files = [f for f in os.listdir(os.path.join(source_config['source_path'],
    'observed_correlations')) ]
    files = [os.path.join(source_config['source_path'],
    'observed_correlations',f) for f in files]
    
   
    step_n = 'step_{}'.format(int(step))
    synth_dir = os.path.join(source_config['source_path'],
    step_n,'corr')
    adj_dir = os.path.join(source_config['source_path'],
    step_n,'adjt')
    
    
   
    if files == []:
        msg = 'No input found!'
        raise ValueError(msg)
    
    i = 0
    with click.progressbar(files,label='Determining adjoint sources...') as bar:
        
        for f in bar:
            
            try: 
                tr_o = read(f)[0]
            except:
                print('\nCould not read data: '+os.path.basename(f))
                i+=1
                continue
            try:
                synth_filename = get_synthetics_filename(os.path.basename(f))
                tr_s = read(os.path.join(synth_dir,synth_filename))[0]
            except:
                print('\nCould not read synthetics: '+os.path.basename(f))
                i+=1
                continue
            tr_s.stats.sac = tr_o.stats.sac.copy() #ToDo handle stats in synth.
            tr_s.data = my_centered(tr_s.data,tr_o.stats.npts)    
           
           
            # Get the adjoint source
            func = af.get_adj_func(mtype)
            try:
                adj_src = Trace(data=func(tr_o,tr_s,**options))
                adj_src.stats.sac = tr_o.stats.sac.copy()
                # Save the adjoint source
                file_adj_src = os.path.join(adj_dir,synth_filename)
                adj_src.write(file_adj_src,format='SAC')
            except:
                pass
            #ToDo think about whether stats are ok
            
             
            
            # step index
            i+=1
    
    


def run_adjointsrcs(source_configfile,measr_configfile,step):
    
    source_config=json.load(open(source_configfile))
    measr_config=json.load(open(measr_configfile))
    
    
    mtype = measr_config['mtype']
    
    # TODo all available misfits --  what parameters do they need (if any.)
    if mtype in ['ln_energy_ratio','energy_diff']:
        g_speed = measr_config['g_speed']
        window_params = {}
        window_params['hw'] = measr_config['window_params_hw']
        window_params['sep_noise'] = measr_config['window_params_sep_noise']
        window_params['win_overlap'] = measr_config['window_params_win_overlap']
        window_params['wtype'] = measr_config['window_params_wtype']
        window_params['causal_side'] = measr_config['window_params_causal']
        window_params['plot'] = False # To avoid plotting the same thing twice
        # ToDo think of a better solution here.
   
        adjointsrcs(source_config,mtype,step,g_speed=g_speed,
        window_params=window_params)
    
        
        
        
