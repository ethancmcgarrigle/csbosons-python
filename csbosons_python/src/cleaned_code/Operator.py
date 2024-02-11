import numpy as np
import yaml
import math
import matplotlib
#matplotlib.rcParams['text.usetex'] = True
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt 
import time
from scipy.fft import fft 
from scipy.fft import ifft
import matplotlib 
import matplotlib.pyplot as plt 
from scipy.stats import sem
import Bosefluid_Model 

## helper function for integrate a field 
def integrate_r_intensive(field):
    N_spatial = len(field)
    result = np.sum(field)/N_spatial
    return result


### Operator parent class and Particle number subclass ####  

class Operator:
  # Class definition for CS field theory for Bose fluid model  

  # Constructor   
  def  __init__(self, name, N_samples): 
    # initialize all the system vars 
    self.name = name 

    # initialize samples 
    self.samples = np.zeros(N_samples + 1, dtype=np.complex_)
    self.samplesSq = None 
    self.avg = None
    self.avgSq = None

  # Parent methods for any scalar operator 
  def update_sample_avg(self, sample_indx):
    self.samples[sample_indx] = self.avg
    self.samplesSq[sample_indx] = self.avgSq


  def reset_operator_avg(self):
    self.avg = 0. + 1j*0.
    self.avgSq = 0. + 1j*0.



class N_Operator(Operator):
  def __init__(self, name, N_samples, _calcSquared):
    super().__init__(name, N_samples) 

    self.isCalcSquared = _calcSquared
    self.value = 0.
    self.avg = 0. 
    self.avgSq = 0. 

    # If sq, allocate memory for Squared tracking  
    if(_calcSquared):
      self.samplesSq = np.zeros(N_samples + 1, dtype=np.complex_)


  def update_operator_instantaneous(self, model, iofreq):
    # expecting phi and phistar CSfield objects          
    phi = model.phi
    phistar = model.phistar
    Vol = model.Volume 
    ntau = len(phi[0,:])
    M = len(phi[:,0])
    rho = np.zeros(M, dtype=np.complex_)
    N = 0. + 1j*0.
    for itau in range(0, int(ntau)):
      itaum1 = ( (int(itau) - 1) % int(ntau) + int(ntau)) % int(ntau) # for PBC 
      rho += phistar[:, itau] * phi[:, itaum1]
    N = integrate_r_intensive(rho/ntau) * Vol
    #self.samples[sample_indx] = N
    self.value = 0.
    self.value += N
    if(self.isCalcSquared):
      N2 = N**2
      #self.samplesSq[sample_indx] = N**2

    # Add this to the average     
    self.avg += N/iofreq 
    self.avgSq += N2/iofreq


  def returnParticleNumber(self):
    return self.value



