import numpy as np

import numpy as np

def sine_taper(k,N):

    """
    Return the sine taper (Riedel & Sidorenko, IEEE'95)
    :type k: int
    :param k: return the k'th taper
    :type N: int
    :param N: Number of samples
    """

    x = np.linspace(0,N+1,N) # make sure it goes to 0

    
    norm = np.sqrt(2.)/np.sqrt(float(N-1))
    y = norm * np.sin(np.pi*k*x/(N+1))
    y[0] = 0.0
    y[-1] = 0.0
   
    return(y)


def choose_basis_function(btype):

    if btype == 'sine_taper':
        func = sine_taper

    else:
        msg = "Currently implemented basis functions:\
 Sine_taper"
        raise ValueError(msg)
    return func