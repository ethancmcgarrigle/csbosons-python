import numpy as np
import yaml
import math
import time
from scipy.fft import fft 
from scipy.fft import ifft
from scipy.stats import sem
from dp1_FFT import *
# Import our custom classes 
#from Bosefluid_Model import Bosefluid_Model
#import Timesteppers 


### Timesteppers ####  

class Timesteppers:
  # Class definition for CS field theory for Bose fluid model  

  # Constructor   

  # ETD method  
  @staticmethod
  def ETD(phi, phistar, _dSdphistar, _dSdphi, _lincoef, _lincoef_phistar, _nonlincoef, _nonlincoef_phistar, noisescl, noisescl_phistar, _CLnoise):
    #print(phi) 
    ntau = len(phi[0, :])
    N_spatial = len(phi)
    # Exponential-Time-Differencing, assumes off-diagonal stepping 
  
    # Function to step phi and phistar with ETD  
    #print(phi)
    phi = fft_dp1(phi) * (_lincoef) 
    #print(phi)
    phistar = fft_dp1(phistar) * (_lincoef_phistar) 
  
    # add nonlinear term, off-diagonal relaxation 
    #print(_nonlincoef) 
    phi += (fft_dp1(_dSdphistar) * _nonlincoef)
    phistar += (fft_dp1(_dSdphi) * _nonlincoef_phistar)
    #print(phi) 
  
    # noise
    _noise = np.zeros((N_spatial, ntau), dtype=np.complex_)
    _noisestar = np.zeros((N_spatial, ntau), dtype=np.complex_)
    _noise.fill(0.) 
    _noisestar.fill(0.) 
  
    if(_CLnoise):
      # ETD assumes off-diagonal stepping, generate nosie and scale  
      _noise += np.random.normal(0, 1., (N_spatial, ntau)) + 1j * np.random.normal(0, 1., (N_spatial, ntau))
      _noisestar += np.conjugate(_noise) 
      # FFT and Scale by fourier coeff 
      _noise = fft_dp1(_noise) * noisescl 
      #_noisestar = fft_dp1(_noisestar) * np.conj(noisescl) 
      _noisestar = fft_dp1(_noisestar) * noisescl_phistar 
      # Add the noise to CS fields  
      phi += _noise 
      phistar += _noisestar 
  
    # inverse fft  
    phi = ifft_dp1(phi) 
    phistar = ifft_dp1(phistar) 

    #print(phi) 
    return [phi, phistar]
    # Return state vector (packaged phi/phistar vector)


  @staticmethod
  def ETD_implicit(phi, phistar, dSdphistar, dSdphi, lincoef, lincoef_phistar, nonlincoef, nonlincoef_phistar, noisescl, noisescl_phistar, _CLnoise, An, _isShifting, _dt):
  
      _tolerance = 1E-5
      max_iters = 1000
      num_iters = 1
      cost = 0.1
      ntau = len(phi[0, :])
      N_spatial = len(phi)
  
      phi_cp = np.zeros((Nx**dim, ntau), dtype=np.complex_)
      phistar_cp = np.zeros((Nx**dim, ntau), dtype=np.complex_)
      tmp = np.zeros((Nx**dim, ntau), dtype=np.complex_)
      tmp2 = np.zeros((Nx**dim, ntau), dtype=np.complex_)
  
      phi_cp += phi
      phistar_cp += phistar
  
      # Need to generate noise and fix it throughout iterations 
      # noise
      noise = np.zeros((N_spatial, ntau), dtype=np.complex_)
      noisestar = np.zeros((N_spatial, ntau), dtype=np.complex_)
  
      noise.fill(0.) 
      noisestar.fill(0.) 
  
      if(_CLnoise):
        # ETD assumes off-diagonal stepping, generate nosie and scale  
        noise = np.random.normal(0, 1., (N_spatial, ntau)) + 1j * np.random.normal(0, 1., (N_spatial, ntau))
        noisestar = np.conj(noise) 
        # FFT and Scale by fourier coeff 
        noise = fft_dp1(noise) 
        noisestar = fft_dp1(noisestar) 
        #noise = fft_dp1(noise) * noisescl 
        #noisestar = fft_dp1(noisestar) * noisescl_phistar 
  
      # do ETD step to start 
      phi, phistar = ETD(phi, phistar, dSdphistar, dSdphi, lincoef, lincoef_phistar, nonlincoef, nonlincoef_phistar, noisescl, noisescl_phistar, False)
      # Add the noise 
      if(_CLnoise):
        phi = fft_dp1(phi) 
        phistar = fft_dp1(phistar) 
        phi += noise 
        phistar += noisestar
        phi = ifft_dp1(phi) 
        phistar = ifft_dp1(phistar) 
  
      while(cost > _tolerance):
        # Calculate forces with new CS fields 
        # Update the nonlinear forces before stepping; fill forces clears dSdphistar and dSdphi 
        fill_forces(phi, phistar, dSdphistar, dSdphi, ntau, _psi, ensemble, _mu, _g, beta, _coeff_phi, _coeff_phistar, _isShifting, An)
        if(_isShifting):
          # Only if _coeff_phi and _coeff_phistar are CL-time independent
          for j in range(0, ntau):
            lincoef[:, j] = An[:, j] + _coeff_phi[:, j]
            # shift any negative real and nonzero imag part away 
            lincoef_phistar[:, j] = np.conj(An[:, j]) + _coeff_phistar[:, j]
            # Correct diverging terms by using Euler limit of ETD
            # Python's FFT accounts for scaling, i.e. ifft(fft(a) == a , therefore, take out the scaling factors  
            for m in range(0, N_spatial): 
              if(lincoef[m, j] == 0.):
                nonlincoef[m, j] = -1. * ntau * _dt 
                nonlincoef_phistar[m, j] = -1. * ntau * _dt 
                noisescl[m, j] = np.sqrt(ntau * _dt)
                noisescl_phistar[m, j] = np.sqrt(ntau * _dt)
                #noisescl[m, j] = np.sqrt(ntau * dt / dV)
                #noisescl_phistar[m, j] = np.sqrt(ntau * dt / dV)
              else: 
                nonlincoef[m, j] = (np.exp(-lincoef[m,j] * ntau * _dt) - 1.)/lincoef[m, j]
                nonlincoef_phistar[m, j] = (np.exp(-lincoef_phistar[m,j] * ntau * _dt) - 1.)/lincoef_phistar[m, j]
                noisescl[m, j] = np.sqrt((1. - np.exp(-2. * lincoef[m, j] * ntau * _dt))/(2. * lincoef[m, j]))
                noisescl_phistar[m, j] = np.sqrt((1. - np.exp(-2. * lincoef_phistar[m, j] * ntau * _dt))/(2. * lincoef_phistar[m, j]))
                #noisescl[m, j] = np.sqrt((1. - np.exp(-2. * lincoef[m, j] * ntau * _dt))/(2. * lincoef[m, j] * dV))
                #noisescl_phistar[m, j] = np.sqrt((1. - np.exp(-2. * lincoef_phistar[m, j] * ntau * _dt))/(2. * lincoef_phistar[m, j] * dV))
            lincoef[:, j] = np.exp(- lincoef[:, j] * ntau * _dt)
            lincoef_phistar[:, j] = np.exp(- lincoef_phistar[:, j] * ntau * _dt)
  
        tmp.fill(0.)
        tmp2.fill(0.)
        tmp += phi
        tmp2 += phistar
        # Reset fields to initial state 
        phi.fill(0.)
        phistar.fill(0.)
        phi += phi_cp
        phistar += phistar_cp

        # step the fields 
        phi, phistar = ETD(phi, phistar, dSdphistar, dSdphi, lincoef, lincoef_phistar, nonlincoef, nonlincoef_phistar, noisescl, noisescl_phistar, False)
        # Add the noise 
        if(_CLnoise):
          phi = fft_dp1(phi) 
          phistar = fft_dp1(phistar) 
          phi += (noise * noisescl) 
          phistar += (noisestar * noisescl_phistar) 
          phi = ifft_dp1(phi) 
          phistar = ifft_dp1(phistar) 
        # prep for cost 
        tmp -= phi 
        tmp2 -= phistar 
  
        cost = 0.
        cost = np.max(np.abs(tmp)) + np.max(np.abs(tmp2))
        num_iters += 1
  
        #print(cost)
        #print(num_iters)
        if(cost < _tolerance):
          #print(num_iters)
          break
  
        if(num_iters > max_iters):
          print('Warning, we have exceeded the max number of iterations!')
          break
  
      return [phi, phistar]
      # Return state vector (packaged phi/phistar vector)

  
  @staticmethod
  def EM_implicit(phi, phistar, dSdphistar, dSdphi, _isOffDiagonal, _CLnoise, dV, dt, linearcoeff, linearcoeff_star):
      _tolerance = 1E-14
      max_iters = 100
      num_iters = 1
      cost = 0.1
      ntau = len(phi[0, :])
      N_spatial = len(phi)
  
      phi_cp = np.zeros((Nx**dim, ntau), dtype=np.complex_)
      phistar_cp = np.zeros((Nx**dim, ntau), dtype=np.complex_)
      _tmp = np.zeros((Nx**dim, ntau), dtype=np.complex_)
      _tmp2 = np.zeros((Nx**dim, ntau), dtype=np.complex_)
  
      phi_cp += phi
      phistar_cp += phistar
  
      # noise
      noise = np.zeros((N_spatial, ntau), dtype=np.complex_)
      noisestar = np.zeros((N_spatial, ntau), dtype=np.complex_)
      noise.fill(0.) 
      noisestar.fill(0.) 
      mobility = ntau
      #mobility = 1. 
      noisescl_scalar = np.sqrt(mobility * dt) 
      #noisescl_scalar = np.sqrt(mobility * dt / dV)
  
      # Do an initial EM step 
      phi -= dSdphistar * mobility * dt 
      phistar -= dSdphi * mobility * dt
      if(_CLnoise): 
        noise = np.random.normal(0, 1., (N_spatial, ntau)) + 1j * np.random.normal(0, 1., (N_spatial, ntau))
        noisestar = np.conj(noise)
        noise *= noisescl_scalar
        noisestar *= noisescl_scalar
        phi += noise 
        phistar += noisestar 
  
      while(cost > _tolerance):
        # Calculate forces with new CS fields 
        # Update the nonlinear forces before stepping; fill forces clears dSdphistar and dSdphi 
        fill_forces(phi, phistar, dSdphistar, dSdphi, ntau, _psi, ensemble, _mu, _g, beta, _coeff_phi, _coeff_phistar, False, tmp)
  
        # recompute total forces 
        # d+1 FFT force container
        dSdphistar = fft_dp1(dSdphistar) 
        dSdphi = fft_dp1(dSdphi) 
        # Add linearized contributions 
        dSdphistar += linearcoeff * fft_dp1(phi)
        dSdphi += linearcoeff_star * fft_dp1(phistar)
    
        # inverse d+1 FFT force container
        dSdphistar = ifft_dp1(dSdphistar) 
        dSdphi = ifft_dp1(dSdphi) 
  
        _tmp.fill(0.)
        _tmp2.fill(0.)
        _tmp += phi
        _tmp2 += phistar
        # Reset fields to initial state 
        phi.fill(0.)
        phistar.fill(0.)
        phi += phi_cp
        phistar += phistar_cp
  
        # Do an EM step, using forces evaluated at phi^{l+1}, starting w. fields at l 
        phi -= dSdphistar * mobility * dt 
        phistar -= dSdphi * mobility * dt
        # Add the noise 
        if(_CLnoise): 
          phi += noise 
          phistar += noisestar 
  
        # prep for cost 
        _tmp -= phi 
        _tmp2 -= phistar 
  
        cost = 0.
        cost = np.max(np.abs(_tmp)) + np.max(np.abs(_tmp2))
        num_iters += 1
  
        #print(cost)
        #print(num_iters)
        if(cost < _tolerance):
          #print(num_iters)
          #print(cost)
          break
  
        if(num_iters > max_iters):
          print('Warning, we have exceeded the max number of iterations!')
          break

      return [phi, phistar]
      # Return state vector (packaged phi/phistar vector)

  @staticmethod
  def EM(phi, phistar, dSdphistar, dSdphi, _isOffDiagonal, _CLnoise, dV, dt):
      # Function to step phi and phistar with ETD  
      ntau = len(phi[0, :])
      N_spatial = len(phi[:, 0])
  
      # noise
      noise = np.zeros((N_spatial, ntau), dtype=np.complex_)
      noisestar = np.zeros((N_spatial, ntau), dtype=np.complex_)
      noise.fill(0.) 
      noisestar.fill(0.) 
      mobility = ntau
      #mobility = 1. 
      noisescl_scalar = np.sqrt(mobility * dt) 
      #noisescl_scalar = np.sqrt(mobility * dt / dV)
  
      if(_isOffDiagonal):
        phi -= dSdphistar * mobility * dt 
        phistar -= dSdphi * mobility * dt
        if(_CLnoise): 
          noise = np.random.normal(0, 1., (N_spatial, ntau)) + 1j * np.random.normal(0, 1., (N_spatial, ntau))
          noisestar = np.conj(noise)
      else:
        noisescl_scalar *= np.sqrt(2.) # Real noise, FDT 
        phi -= dSdphi * mobility * dt 
        phistar -= dSdphistar * mobility * dt
        # For diagonal stepping, generate real noise  
        if(_CLnoise): 
          noise = np.random.normal(0, 1., (N_spatial, ntau))  
          noisestar = np.random.normal(0, 1., (N_spatial, ntau)) 
  
  
      # Scale and Add the noise 
      if(_CLnoise): 
        noise *= noisescl_scalar
        noisestar *= noisescl_scalar
  
        phi += noise 
        phistar += noisestar 
  
      return [phi, phistar]
  
