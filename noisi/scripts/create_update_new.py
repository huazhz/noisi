import numpy as np
import pandas as pd
import os
import json
import h5py

from glob import glob
from noisi.util.corr_pairs import glob_obs_corr


def update_sourcemodel(weighted_grad,new_sourcemodel,update_scheme,step_length):
    

    if update_scheme == 'steepest_descent':
        f = h5py.File(new_sourcemodel,'r+')
        f['model'][:] -= step_length * weighted_grad
        f.close()
    return()



def prepare_test_steplength(msrfile,source_config,newdir,min_snr,min_stck,
    data_selection_scheme,select_nr,weights):
    
    obs_dir = os.path.join(source_config['source_path'],'observed_correlations')

    # Read in the csv files of measurement.
    data = pd.read_csv(msrfile[0])
    data.l2_norm.values *= weights[0]
    for i in range[1,len(msrfile)]:
        # We get the addition of both datasets, which means that l2_norms of all
        # measurements are added up and the stations pairs with max overall misfit are chosen
        data.l2_norm += pd.read_csv(msrfile[i]).l2_norm.values *weights[i]
        data.nstack += pd.read_csv(msrfile[i]).nstack.values

    print(data)
    # Get a set of n randomly chosen station pairs. Criteria: minimum SNR, 
    # ---> prelim_stations.txt

    data_accept = pd.merge(data[data.snr >= min_snr],
        data[data.snr_a >= min_snr],how="outer")
    if len(data_accept) == 0:
        raise ValueError('No data match selection criteria.')
    
    data_accept = data[(data.nstack >= min_stck)]
    if len(data_accept) == 0:
        raise ValueError('No data match selection criteria.')
    
    data_accept = data_accept[~(data_accept.l2_norm.apply(np.isnan))]
    if len(data_accept) == 0:
        raise ValueError('No data match selection criteria.')

    print(data_accept)
    # select data...
    if data_selection_scheme =='random':
        data_select = data_accept.sample(n=select_nr)
    elif data_selection_scheme == 'max':
        data_select1 = data_accept.sort_values(by='l2_norm',na_position='first')
        data_select = data_select1.iloc[-select_nr:]


    for i in data_select.index:
        # copy the relevant observed correlation
        obs_dir = os.path.join(source_config['source_path'],
            'observed_correlations')
        sta1 = data_select.at[i,'sta1'].split('.')[0:4]
        sta2 = data_select.at[i,'sta2'].split('.')[0:4]
        obs_correlation = glob_obs_corr('{}.{}.{}.{}'.format(*sta1),
            '{}.{}.{}.{}'.format(*sta2),obs_dir,ignore_network=True)
        

        if len(obs_correlation) > 0:

            # Use preferentially '', '00' channels.
            obs_correlation.sort()
            corr = obs_correlation[0]
            os.system('cp {} {}'.format(corr,os.path.join(newdir,'obs_slt')))
        
    # save metadata
    data_accept.to_csv(os.path.join(newdir,'dat_for_steptest.cumulativeL2.csv'))
    os.system('cp {} {}'.format(os.path.join(source_config['source_path'],
        'inverse_config.json'),newdir))
    return()



############ Preparation procedure #################################################
#prepare_test_steplength = False
# where is the measurement database located?
def create_update(source_model,oldstep,step_length):
    step_length = float(step_length)
    source_config=json.load(open(source_model))
    datadir = os.path.join(source_config['source_path'],'step_' + str(oldstep))
    measr_config = json.load(open(os.path.join(source_config['source_path'],
        'measr_config.json')))
    grad = os.path.join(datadir,'grad','grad_smooth.npy')
    grad = np.load(grad)
    weights = measr_config['weights']
    weighted_grad = np.zeros(grad[0,:,:].shape)
    for i in range(len(weights)):
        weighted_grad += grad[i,:,:]*weights[i]

    msrfile = glob(os.path.join(datadir,"{}.*.measurement.csv".
        format(measr_config['mtype'])))


    # inversion settings
    inv_config = json.load(open(os.path.join(source_config['source_path'],
        'inverse_config.json')))
    min_snr = inv_config['min_snr']
    min_stck = inv_config['min_stack_length']
    update_scheme = inv_config['update_scheme']
    data_selection_scheme = inv_config['data_selection_for_linesearch']
    select_nr = inv_config["nr_measurements_for_linesearch"]

    # Initialize the new step directory
    newstep = int(oldstep) + 1
    newdir = os.path.join(source_config['source_path'],'step_' + str(newstep))

    if not os.path.exists(newdir):
        os.mkdir(newdir)
        os.mkdir(os.path.join(newdir,'obs_slt'))
        os.mkdir(os.path.join(newdir,'corr'))
        os.mkdir(os.path.join(newdir,'adjt'))
        os.mkdir(os.path.join(newdir,'grad'))
        os.mkdir(os.path.join(newdir,'kern'))
        os.system('cp {} {}'.format(os.path.join(source_config['source_path'],
        'inverse_config.json'),newdir))
        os.system('cp {} {}'.format(os.path.join(source_config['source_path'],
        'measr_config.json'),newdir))
        os.system('cp {} {}'.format(source_model,newdir))
        
        prepare_test_steplength(msrfile,source_config,newdir,min_snr,min_stck,
            data_selection_scheme,select_nr,weights)


    os.mkdir(os.path.join(newdir,'steptest_'+str(step_length)))
    os.system('cp {} {}'.format(os.path.join(datadir,'starting_model.h5'),newdir))
    new_sourcemodel = os.path.join(newdir,'starting_model.h5')

    update_sourcemodel(weighted_grad,new_sourcemodel,update_scheme,step_length)

    return()