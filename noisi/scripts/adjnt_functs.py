import numpy as np
from math import pi
#from noisi.scripts import measurements as rm
from noisi.util import windows as wn
from scipy.signal import fftconvolve
from scipy.signal import hilbert




def log_en_ratio_adj(corr_o,corr_s,g_speed,window_params):

    success = False

    window = wn.get_window(corr_o.stats,g_speed,window_params)
    win = window[0]
    msr_o = rm.log_en_ratio(corr_o,g_speed,window_params)
    msr_s = rm.log_en_ratio(corr_s,g_speed,window_params)
    data = wn.my_centered(corr_s.data,corr_o.stats.npts)

    if window[2] == True:
        sig_c = corr_s.data * win
        sig_a = corr_s.data * win[::-1]
        E_plus = np.trapz(np.power(sig_c,2))*corr_s.stats.delta
        E_minus = np.trapz(np.power(sig_a,2))*corr_s.stats.delta
        # to win**2
        u_plus = sig_c * win
        u_minus = sig_a * win[::-1]
        #adjt_src = 2./pi * (msr_s-msr_o) * (u_plus / E_plus - u_minus / E_minus)
        # I don't know where that factor 1/pi came from. Not consistent with new derivation of kernels
        adjt_src = 2. * (msr_s-msr_o) * (u_plus / E_plus - u_minus / E_minus)
        success = True
    else:
        adjt_src = win-win+np.nan
    return adjt_src, success

def windowed_waveform(corr_o,corr_s,g_speed,window_params):
    success = False
    window = wn.get_window(corr_o.stats,g_speed,window_params)
    win = window[0] + window[0][::-1]
    if window[2]:

        u_s = np.multiply(win,corr_s.data)
        u_o = np.multiply(win,corr_o.data)

        adjt_src = np.multiply(win,(u_s-u_o))
        success = True
    else:
        adjt_src = win-win+np.nan

    return adjt_src, success


def square_envelope(corr_o,corr_s,g_speed,
    window_params):
    success = False
    env_s = corr_s.data**2 + np.imag(hilbert(corr_s.data))**2
    env_o = corr_o.data**2 + np.imag(hilbert(corr_o.data))**2
    d_env_1 =  2. * corr_s.data 
    d_env_2 =  (2. * np.imag(hilbert(corr_s.data)))

    u1 = (env_s - env_o) * d_env_1
    u2 = np.imag(hilbert((env_s - env_o) * d_env_2))

    adjt_src = u1 - u2
    
    success = True
    return adjt_src, success




def energy(corr_o,corr_s,g_speed,window_params):

    success = False



    window = wn.get_window(corr_o.stats,g_speed,window_params)
    win = window[0]

    E_plus = np.trapz((corr_s.data * win)**2)
    E_plus_o = np.trapz((corr_o.data * win)**2)

    E_minus = np.trapz((corr_s.data * win[::-1])**2)
    E_minus_o  =  np.trapz((corr_o.data * win[::-1])**2)

    

    if window[2]:
        u1 = 2* np.multiply(np.power(win,2),corr_s.data) * \
        (E_plus-E_plus_o)
        u2 = 2* np.multiply(np.power(win[::-1],2),corr_s.data) * \
        (E_minus-E_minus_o)
        adjt_src = [u1,u2]
        success = True
    else:
        adjt_src = [win-win+np.nan,win-win+np.nan]

    return adjt_src, success




def get_adj_func(mtype):
    if mtype == 'ln_energy_ratio':
        func = log_en_ratio_adj

    elif mtype == 'energy_diff':
        func = energy

    elif mtype == 'windowed_waveform':
        func = windowed_waveform

    elif mtype == 'square_envelope':
        func = square_envelope

    else:
        msg = 'Measurement functional %s not currently implemented.' %mtype
        raise ValueError(msg)
    return func
