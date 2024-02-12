import numpy as np
import math
from scipy.fft import fft 
from scipy.fft import ifft

# d+1 Fourier Transforms --- passed test check for normalization  
def fft_dp1(_CSfield, _dontScale=True):
    # d+1 Fourier transform of a CS field object -- 2D numpy array \in C^ (Nx**dim , ntau)
    ntau = len(_CSfield[0,:])
    N_spatial = len(_CSfield)

    # Spatial Fourier transform -- column-by-column i.e. each tau slice is FFT'd 
    for j in range(0, ntau):
      _CSfield[:,j] = fft(_CSfield[:, j])

    # Now do the Fourier transform in imaginary time to Matsubara freq.

    for m in range(0, N_spatial):
      _CSfield[m,:] = fft(_CSfield[m, :])

    return _CSfield


def ifft_dp1(_CSfield):
    # d+1 Fourier transform of a CS field object -- 2D numpy array \in C^ (Nx**dim , ntau)
    ntau = len(_CSfield[0,:])
    N_spatial = len(_CSfield)

    # Spatial Fourier transform -- column-by-column i.e. each tau slice is FFT'd 
    for j in range(0, ntau):
      #_CSfield[:,j] = ifft(_CSfield[:, j]) * N_spatial
      _CSfield[:,j] = ifft(_CSfield[:, j]) 
 
    # Now do the Fourier transform in imaginary time to Matsubara freq.

    for m in range(0, N_spatial):
      #_CSfield[m,:] = ifft(_CSfield[m, :]) * ntau
      _CSfield[m,:] = ifft(_CSfield[m, :])

    return _CSfield

