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

  # Euler Maruyama
  @staticmethod
  def EM(phi, phistar, dSdphistar, dSdphi, _CLnoise, dV, dt):
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
      #noisescl_scalar = np.sqrt(mobility * dt) 
      noisescl_scalar = np.sqrt(mobility * dt / dV)

      # Off-diagonal stepping, e.g. phi is stepped using dS/dphistar   
      phi -= dSdphistar * mobility * dt 
      phistar -= dSdphi * mobility * dt
      if(_CLnoise): 
        noise = np.random.normal(0, 1., (N_spatial, ntau)) + 1j * np.random.normal(0, 1., (N_spatial, ntau))
        noisestar = np.conj(noise)
  
  
      # Scale and Add the noise 
      if(_CLnoise): 
        noise *= noisescl_scalar
        noisestar *= noisescl_scalar
  
        phi += noise 
        phistar += noisestar 
  
      return [phi, phistar]
  

  # ETD method  
  @staticmethod
  def ETD(phi, phistar, _dSdphistar, _dSdphi, _lincoef, _lincoef_phistar, _nonlincoef, _nonlincoef_phistar, noisescl, noisescl_phistar, _CLnoise):
    #print(phi) 
    ntau = len(phi[0, :])
    N_spatial = len(phi)
    # Exponential-Time-Differencing, assumes off-diagonal stepping 
  
    # Function to step phi and phistar with ETD  
    phi = fft_dp1(phi) * (_lincoef) 
    phistar = fft_dp1(phistar) * (_lincoef_phistar) 
  
    # add nonlinear term, off-diagonal relaxation 
    phi += (fft_dp1(_dSdphistar) * _nonlincoef)
    phistar += (fft_dp1(_dSdphi) * _nonlincoef_phistar)
  
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


 #  @staticmethod
 #  def ETD_implicit(model, driver, phi, phistar, dSdphistar, dSdphi, lincoef, lincoef_phistar, nonlincoef, nonlincoef_phistar, noisescl, noisescl_phistar, _CLnoise, _isShifting, _dt):
 #  
 #      _tolerance = 1E-6
 #      max_iters = 1000
 #      num_iters = 1
 #      cost = 0.1
 #      ntau = len(phi[0, :])
 #      N_spatial = len(phi)
 #  
 #      phi_cp = np.zeros((N_spatial, ntau), dtype=np.complex_)
 #      phistar_cp = np.zeros((N_spatial, ntau), dtype=np.complex_)
 #      tmp = np.zeros((N_spatial, ntau), dtype=np.complex_)
 #      tmp2 = np.zeros((N_spatial, ntau), dtype=np.complex_)
 #  
 #      phi_cp += phi
 #      phistar_cp += phistar
 #  
 #      # Need to generate noise and fix it throughout iterations 
 #      # noise
 #      noise = np.zeros((N_spatial, ntau), dtype=np.complex_)
 #      noisestar = np.zeros((N_spatial, ntau), dtype=np.complex_)
 #  
 #      noise.fill(0.) 
 #      noisestar.fill(0.) 
 #  
 #      if(_CLnoise):
 #        # ETD assumes off-diagonal stepping, generate noise 
 #        noise = np.random.normal(0, 1., (N_spatial, ntau)) + 1j * np.random.normal(0, 1., (N_spatial, ntau))
 #        noisestar = np.conj(noise) 
 #        # FFt the noise and store it for implicit iteration; don't scale since the tstepcoefficient may change at each timestep 
 #        noise = fft_dp1(noise) 
 #        noisestar = fft_dp1(noisestar) 
 #  
 #      # do ETD step to start 
 #      phi, phistar = Timesteppers.ETD(phi, phistar, dSdphistar, dSdphi, lincoef, lincoef_phistar, nonlincoef, nonlincoef_phistar, noisescl, noisescl_phistar, False)
 #
 #      # Add the noise 
 #      if(_CLnoise):
 #        phi = fft_dp1(phi) 
 #        phistar = fft_dp1(phistar) 
 #        phi += (noise * noisescl) 
 #        phistar += (noisestar * noisescl_phistar)
 #        phi = ifft_dp1(phi) 
 #        phistar = ifft_dp1(phistar) 
 #  
 #      while(cost > _tolerance):
 #        # Calculate forces with new CS fields 
 #        # Update the nonlinear forces before stepping; fill forces clears dSdphistar and dSdphi and calculates B(n) 
 #        model.fill_forces()
 #
 #        # Refill the linear tstep coefficients if we shift the linear force  
 #        if(_isShifting):
 #          driver.fill_tstep_coefs()
 #  
 #        tmp.fill(0.)
 #        tmp2.fill(0.)
 #        tmp += phi
 #        tmp2 += phistar
 #        # Reset fields to initial state 
 #        phi.fill(0.)
 #        phistar.fill(0.)
 #        phi += phi_cp
 #        phistar += phistar_cp
 #
 #        # step the fields 
 #        phi, phistar = Timesteppers.ETD(phi, phistar, model.dSdphistar, model.dSdphi, model.lincoef, model.lincoef_phistar, driver.nonlincoef, driver.nonlincoef_phistar, driver.noisescl, driver.noisescl_phistar, False)
 #        # Add the noise 
 #        if(_CLnoise):
 #          phi = fft_dp1(phi) 
 #          phistar = fft_dp1(phistar) 
 #          phi += (noise * driver.noisescl) 
 #          phistar += (noisestar * driver.noisescl_phistar) 
 #          phi = ifft_dp1(phi) 
 #          phistar = ifft_dp1(phistar) 
 #        # prep for cost 
 #        tmp -= phi 
 #        tmp2 -= phistar 
 #
 #        model.phi = phi  
 #        model.phistar = phistar  
 #  
 #        cost = 0.
 #        cost = np.max(np.abs(tmp)) + np.max(np.abs(tmp2))
 #        num_iters += 1
 #  
 #        #print(cost)
 #        #print(num_iters)
 #        if(cost < _tolerance):
 #          #print(num_iters)
 #          break
 #  
 #        if(num_iters > max_iters):
 #          print('Warning, we have exceeded the max number of iterations!')
 #          break
 #  
 #      return [phi, phistar]
 #      # Return state vector (packaged phi/phistar vector)
 #
 #  
 #  @staticmethod
 #  def EM_implicit(model, driver, phi, phistar, dSdphistar, dSdphi, _CLnoise, dV, dt, _isShifting):
 #      _tolerance = 1E-6
 #      max_iters = 500
 #      num_iters = 1
 #      cost = 0.1
 #      ntau = len(phi[0, :])
 #      N_spatial = len(phi)
 #  
 #      phi_cp = np.zeros((N_spatial, ntau), dtype=np.complex_)
 #      phistar_cp = np.zeros((N_spatial, ntau), dtype=np.complex_)
 #      _tmp = np.zeros((N_spatial, ntau), dtype=np.complex_)
 #      _tmp2 = np.zeros((N_spatial, ntau), dtype=np.complex_)
 #  
 #      phi_cp += phi
 #      phistar_cp += phistar
 #  
 #      # noise
 #      noise = np.zeros((N_spatial, ntau), dtype=np.complex_)
 #      noisestar = np.zeros((N_spatial, ntau), dtype=np.complex_)
 #      noise.fill(0.) 
 #      noisestar.fill(0.) 
 #      mobility = ntau
 #      #mobility = 1. 
 #      #noisescl_scalar = np.sqrt(mobility * dt) 
 #      noisescl_scalar = np.sqrt(mobility * dt / dV)
 #  
 #      # Do an initial EM step 
 #      phi -= dSdphistar * mobility * dt 
 #      phistar -= dSdphi * mobility * dt
 #      if(_CLnoise):
 #        # Generate noise for iteration  
 #        noise = np.random.normal(0, 1., (N_spatial, ntau)) + 1j * np.random.normal(0, 1., (N_spatial, ntau))
 #        noisestar = np.conj(noise)
 #        noise *= noisescl_scalar
 #        noisestar *= noisescl_scalar
 #        phi += noise 
 #        phistar += noisestar 
 #  
 #      while(cost > _tolerance):
 #        # Calculate forces with new CS fields 
 #        # Update the nonlinear forces before stepping; fill forces clears dSdphistar and dSdphi 
 #        model.fill_forces()
 #
 #        # Refill the linear tstep coefficients if we shift the linear force  
 #        if(_isShifting):
 #          model.fill_tstep_coefs()
 #  
 #        # recompute total forces 
 #        # d+1 FFT force container
 #        dSdphistar = fft_dp1(model.dSdphistar) 
 #        dSdphi = fft_dp1(model.dSdphi) 
 #        # Add linearized contributions 
 #        dSdphistar += (model.lincoef + model.Bn) * fft_dp1(phi)
 #        dSdphi += (model.lincoef_phistar + model.Bn_star) * fft_dp1(phistar)
 #    
 #        # inverse d+1 FFT force container
 #        dSdphistar = ifft_dp1(dSdphistar) 
 #        dSdphi = ifft_dp1(dSdphi) 
 #  
 #        _tmp.fill(0.)
 #        _tmp2.fill(0.)
 #        _tmp += phi
 #        _tmp2 += phistar
 #        # Reset fields to initial state 
 #        phi.fill(0.)
 #        phistar.fill(0.)
 #        phi += phi_cp
 #        phistar += phistar_cp
 #  
 #        # Do an EM step, using forces evaluated at phi^{l+1}, starting w. fields at l 
 #        # Take EM step 
 #        phi, phistar = Timesteppers.EM(phi, phistar, dSdphistar, dSdphi, False, dV, dt)
 #
 #        # Add the noise; it's already scaled  
 #        if(_CLnoise): 
 #          phi += noise 
 #          phistar += noisestar 
 #  
 #        # prep for cost 
 #        _tmp -= phi 
 #        _tmp2 -= phistar 
 #
 #        model.phi = phi  
 #        model.phistar = phistar  
 #        cost = 0.
 #        cost = np.max(np.abs(_tmp)) + np.max(np.abs(_tmp2))
 #        num_iters += 1
 #  
 #        #print(cost)
 #        #print(num_iters)
 #        if(cost < _tolerance):
 #          #print(num_iters)
 #          #print(cost)
 #          break
 #  
 #        if(num_iters > max_iters):
 #          print('Warning, we have exceeded the max number of iterations!')
 #          break
 #
 #      return [phi, phistar]
 #      # Return state vector (packaged phi/phistar vector)

