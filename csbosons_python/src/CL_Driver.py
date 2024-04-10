import numpy as np
import yaml
import math
import matplotlib
#matplotlib.rcParams['text.usetex'] = True ## Use this on mac osx?
matplotlib.use('TkAgg') # use this on linux 
import matplotlib.pyplot as plt 
import time
from scipy.fft import fft 
from scipy.fft import ifft
import matplotlib 
import matplotlib.pyplot as plt 
from scipy.stats import sem
# Import our custom classes 
from dp1_FFT import *
from Bosefluid_Model import Bosefluid_Model
from Timesteppers import Timesteppers 
from Operator import N_Operator


### Driver Class ####  

class CL_Driver:
  # Class definition for CS field theory for Bose fluid model  

  # Constructor   
  def  __init__(self, Model, CLnoise, ETD, isShifting, dt, numtsteps, iofreq, num_samples): 
    # initialize all the system vars 
    self.CLnoise = CLnoise
    self.ETD = ETD 
    #self.do_implicit = do_implicit
    self.isShifting = isShifting
    self.dt = dt
    self.numtsteps = numtsteps
    self.iofreq = iofreq 
    self.num_samples = num_samples
    self.model = Model

    ## Access the grid parameters from the model  
    self.Nx = Model.Nx
    self.dim = Model.dim
    self.dV = Model.dV
    self.Volume = Model.Volume
    self.N_spatial = Model.N_spatial
    self.ntau = Model.ntau
    self.dtau = Model.dtau

    # d+1 dim fields: use 2D np arrays: (Nx **d) x Ntau 
    # initialize CS fields at zero as default 
    # Noise, nonlinearcoefs, linear_tstep_coefs initialize 
    self.noise = np.zeros((self.Nx**self.dim, self.ntau), dtype=np.complex_)
    self.noisestar = np.zeros((self.Nx**self.dim, self.ntau), dtype=np.complex_)
    self.lin_tstep_coef = np.zeros((self.Nx**self.dim, self.ntau), dtype=np.complex_)
    self.linstar_tstep_coef= np.zeros((self.Nx**self.dim, self.ntau), dtype=np.complex_)
    self.nonlincoef = np.zeros((self.Nx**self.dim, self.ntau), dtype=np.complex_)
    self.nonlincoef_phistar = np.zeros((self.Nx**self.dim, self.ntau), dtype=np.complex_)
    self.noisescl = np.zeros((self.Nx**self.dim, self.ntau), dtype=np.complex_)
    self.noisescl_phistar = np.zeros((self.Nx**self.dim, self.ntau), dtype=np.complex_)

    # Print the simulation setup 
    self.print_model_startup()

    # Model has been created, linear and nonlinear forces have been filled. Need to fill the tstep coefs 
    self.fill_tstep_coefs()

    # Need to create the operators (N_operator) 
    self.model.N_operator = N_Operator('N', num_samples, True) # Particle number operator. True will track N^2 as well.  
    # Fill first sample 
    #print(self.model.N_operator.returnParticleNumber())
    self.model.N_operator.samples[0] = self.model.N_operator.returnParticleNumber()
    #print(self.model.N_operator.samples[0])
    self.model.N_operator.samplesSq[0] = self.model.N_operator.returnParticleNumber() ** 2

    self.model.N_operator.update_operator_instantaneous(self.model, iofreq) 

    # Setup CL time Simulation time 
    self.t_s = np.zeros(num_samples + 1)
    # initialize the fictitious time 
    self.t = 0.

  
  def print_model_startup(self):
      print()
      print()
      print('-----Bosefluid Simulation: Bosonic Coherent States----')
      print()
      print()
      print('Ensemble: ' + self.model.ensemble)
      print()
      if(self.model.ensemble == 'GRAND'):
          print('Chemical Potential: ' + str(self.model.mu))
      else:
          # CE (CANONICAL)
          print('N constraint: ' + str(self.model.N_input))
          print()
          print('lambda_psi Mobility: ' + str(self.model.lambda_psi))
      
      print()
      print()
      print('Pair Repulsion Potential Strength: ' + str(self.model.g))
      print()
      print('Temperature : ' + str(1/self.model.beta) + ' Kelvin')
      print()
      print('Running for ' + str(self.numtsteps) + ' timesteps')
      print()
      print('Using Ntau = ' + str(self.model.ntau) + ' tau slices' )
      print()
      print('Using Nx = ' + str(self.model.Nx) + ' grid points per dimension' )
      print()
      print('Using L = ' + str(self.model.L) + ' grid length per dimension' )
      print()
      print('Volume = ' + str(self.model.Volume))
      print()
      print('dV = ' + str(self.model.dV))
      print()
      print()
      print('Complex Langevin Sampling')
      print()
      print(' Using timestep: ' + str(self.dt))
      print()
      print(' Using Noise? ' + str(self.CLnoise))
      print()


  def fill_tstep_coefs(self):
    # We need coefficients that are a function of the linear coefficient and the timestep to multiply the forces
    self.lin_tstep_coef.fill(0.)
    self.linstar_tstep_coef.fill(0.)
    # Loop through imaginary time points 
    for j in range(0, self.ntau):
      self.lin_tstep_coef[:,j] = self.model.lincoef[:, j] + self.model.Bn[:, j] # shift by optional B(n)
      self.linstar_tstep_coef[:,j] = np.conj(self.lin_tstep_coef[:,j])
      # Need to correct diverging elements in space (k=0, etc.) 
      for m in range(0, self.N_spatial): 
        if(self.lin_tstep_coef[m, j] == 0.):
          self.nonlincoef[m, j] = -1. * self.ntau * self.dt 
          self.nonlincoef_phistar[m, j] = -1. * self.ntau * self.dt 
          #self.noisescl[m, j] = np.sqrt(self.ntau * self.dt)
          #self.noisescl_phistar[m, j] = np.sqrt(self.ntau * self.dt)
          self.noisescl[m, j] = np.sqrt(self.ntau * self.dt / self.dV)
          self.noisescl_phistar[m, j] = np.sqrt(self.ntau * self.dt / self.dV)
        else: 
          self.nonlincoef[m, j] = (np.exp(-self.lin_tstep_coef[m,j] * self.ntau * self.dt) - 1.)/self.lin_tstep_coef[m, j]
          self.nonlincoef_phistar[m, j] = (np.exp(-self.linstar_tstep_coef[m,j] * self.ntau * self.dt) - 1.)/self.linstar_tstep_coef[m, j]
          self.noisescl[m, j] = np.sqrt((1. - np.exp(-2. * self.lin_tstep_coef[m, j] * self.ntau * self.dt))/(2. * self.lin_tstep_coef[m, j] * self.dV))
          self.noisescl_phistar[m, j] = np.sqrt((1. - np.exp(-2. * self.linstar_tstep_coef[m, j] * self.ntau * self.dt))/(2. * self.linstar_tstep_coef[m, j] * self.dV))
      self.lin_tstep_coef[:, j] = np.exp(- self.lin_tstep_coef[:, j] * self.ntau * self.dt)
      self.linstar_tstep_coef[:, j] = np.exp(- self.linstar_tstep_coef[:, j] * self.ntau * self.dt)


  def run_simulation(self, _isPlotting):
    # Main code for running CL simulation 
    start = time.time()
    ctr = 1
    N = 0. + 1j*0.
    # main loop for CL iterations 
    for l in range(0, self.numtsteps + 1):
      # Update the nonlinear forces before stepping; fill forces refreshes dSdphistar and dSdphi 
      self.model.fill_forces()  # creates a shift in linear_coef if we are shifting  
  
      # Refill the linear tstep coefficients if we shift the linear force  
      if(self.isShifting):
        self.fill_tstep_coefs()

      # Perform the timestep 
      self.timestep()

      # ---- Calculate and Update Observables ------- 
      # Check for divergence 
      N = self.model.N_operator.returnParticleNumber()
      if(np.isnan(N)):
        print('Trajectory diverged at iteration: ' + str(l) + ' and CL time = ' + str(self.t))
        break
     
      self.t += self.dt

      self.model.N_operator.update_operator_instantaneous(self.model, self.iofreq) 
    
      # Output on interval
      if(l % self.iofreq == 0 and l > 0):
         if(ctr %  25):
           print("Completed {} of {} steps".format(l, self.numtsteps))
         # opout.write("{} {} {} {} {}\n".format(it, Msum.real / Navg, Msum.imag / Navg, psi.real, psi.imag))
         
         self.t_s[ctr] = self.t
         # update the sample avg. 
         self.model.N_operator.update_sample_avg(ctr)
         # clear the averages 
         self.model.N_operator.reset_operator_avg()
    
         ctr += 1
      
  
    end = time.time()
    print()
    print()
    if(l == self.numtsteps):
      print('Simulation finished: Runtime = ' + str(end - start) + ' seconds')
    
    # Print the results (noise long-time averages)
    print()
    print()
    print('The Boson Particle Number is: ' + str(np.mean(self.model.N_operator.samples[4:ctr].real)))
    #print('The Boson Particle Number is: ' + str(self.model.N_operator.returnParticleNumber()))
    print()
    print('The Particle Number squared is: ' + str(np.mean(self.model.N_operator.samplesSq[4:ctr].real)))
    print()
    print('The density is: ' + str(np.mean(self.model.N_operator.samples[4:ctr].real)/self.model.Volume) + ' plus/minus ' + str(sem(self.model.N_operator.samples[10:ctr].real)/self.model.Volume))
    print()

    if(_isPlotting):
      self.plot_results(ctr)

  

  def timestep(self):
      _ETD = self.ETD
      #_do_implicit = self.do_implicit
      # For ETD, can  proceed; for EM, must add linear force contributions to the conatiners  
      if(not _ETD):
        # d+1 FFT force container
        self.model.dSdphistar = fft_dp1(self.model.dSdphistar) 
        self.model.dSdphi = fft_dp1(self.model.dSdphi) 
        # Add linearized contributions
        self.model.dSdphistar += (self.model.lincoef + self.model.Bn) * fft_dp1(self.model.phi)
        #dSdphi += np.conj(tmp) * fft_dp1(phistar)
        self.model.dSdphi += (self.model.lincoef_phistar + self.model.Bn_star) * fft_dp1(self.model.phistar)
    
        # need to iFFT phi and phistar
        self.model.phi = ifft_dp1(self.model.phi) 
        self.model.phistar = ifft_dp1(self.model.phistar)
    
        # inverse d+1 FFT force container
        self.model.dSdphistar = ifft_dp1(self.model.dSdphistar) 
        self.model.dSdphi = ifft_dp1(self.model.dSdphi) 
    
        # Do EM step 
 #        if(_do_implicit):
 #          self.model.phi, self.model.phistar = Timesteppers.EM_implicit(self.model, self, self.model.phi, self.model.phistar, self.model.dSdphistar, self.model.dSdphi, self.CLnoise, self.model.dV, self.dt, self.isShifting) 
        #else:
        self.model.phi, self.model.phistar = Timesteppers.EM(self.model.phi, self.model.phistar, self.model.dSdphistar, self.model.dSdphi, self.CLnoise, self.dV, self.dt)
      else:  
 #        if(_do_implicit): 
 #          self.model.phi, self.model.phistar = Timesteppers.ETD_implicit(self.model, self, self.model.phi, self.model.phistar, self.model.dSdphistar, self.model.dSdphi, self.lin_tstep_coef, self.linstar_tstep_coef, self.nonlincoef, self.nonlincoef_phistar, self.noisescl, self.noisescl_phistar, self.CLnoise, self.isShifting, self.dt)
        #else:
        self.model.phi, self.model.phistar = Timesteppers.ETD(self.model.phi, self.model.phistar, self.model.dSdphistar, self.model.dSdphi, self.lin_tstep_coef, self.linstar_tstep_coef, self.nonlincoef, self.nonlincoef_phistar, self.noisescl, self.noisescl_phistar, self.CLnoise)
  



  def plot_results(self, ctr):  
    # plot the results
    plt.style.use('./plot_style.txt')
    plt.figure(figsize=(6,6))
    plt.title('Particle Number: CL Simulation', fontsize = 20, fontweight = 'bold')
    plt.plot(self.t_s[0:ctr], self.model.N_operator.samples[0:ctr].real, '*-', color = 'green', linewidth = 0.5, label = 'real part')
    plt.plot(self.t_s[0:ctr], self.model.N_operator.samples[0:ctr].imag, '*-', color = 'skyblue', linewidth=0.5,label = 'imaginary part')
    #plt.plot(t_s, np.ones(len(t_s)), 'k', label = 'Constraint')
    plt.xlabel('CL time', fontsize = 16, fontweight = 'bold')
    plt.ylabel('$N$', fontsize = 20, fontweight = 'bold') 
    plt.ylim([-5, 25])
    plt.legend()
    plt.show()
  
  

    plt.figure(figsize=(6,6))
    plt.title('$N^2$ : CL Simulation', fontsize = 20, fontweight = 'bold')
    plt.plot(self.t_s[0:ctr], self.model.N_operator.samplesSq[0:ctr].real, '*-', color = 'purple', label = 'Real part')
    plt.plot(self.t_s[0:ctr], self.model.N_operator.samplesSq[0:ctr].imag, '*-', color = 'skyblue', label = 'Imaginary part')
    # plt.plot(t_s, np.ones(len(t_s)), 'k', label = 'Constraint')
    plt.xlabel('CL time', fontsize = 16, fontweight = 'bold')
    plt.ylabel('$N^2$', fontsize = 20, fontweight = 'bold') 
    #plt.ylim([-5, 20])
    plt.legend()
    plt.show()
  
