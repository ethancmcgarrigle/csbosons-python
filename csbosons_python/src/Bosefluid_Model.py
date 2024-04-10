import numpy as np
import yaml
import math
import matplotlib
#matplotlib.rcParams['text.usetex'] = True
#matplotlib.use('TkAgg')
#import matplotlib.pyplot as plt 
import time
from scipy.fft import fft 
from scipy.fft import ifft
import matplotlib 
import matplotlib.pyplot as plt 
from dp1_FFT import *
from scipy.stats import sem
from Operator import N_Operator

### Bose fluid Model class ####  

class Bosefluid_Model:
  # Class definition for CS field theory for Bose fluid model  

  # Constructor   
  def  __init__(self, mu, g, beta, _lambda, dim, L, ntau, ensemble, Nx, isShifting):
    # initialize all the system vars 
    self.mu = mu
    self.g = g 
    self.L = L
    self.beta = beta
    self.dim = dim
    self.ntau = ntau
    self.ensemble = ensemble 
    self.Nx = Nx 
    self._lambda = _lambda
    self.lambda_psi = 0.
    if(ensemble == 'GRAND'):
      self.N_input = None
      self.psi = None
    else:
      self.lambda_psi = 1.0
      self.N_input = 25. 
      self.psi = 0. + 1j*0. 

    isPsizero = False 
    self.isShifting = isShifting

    ## Derived variables 
    # Space grid 
    self.Volume = L**dim
    self.N_spatial = Nx**dim
    self.dV = self.Volume/(Nx**dim) 
    # Imaginary time discretization; fields obey PBC in imaginary time dimension 
    self.dtau = beta/ntau 

    # d+1 dim fields: use 2D np arrays: (Nx **d) x Ntau; i.e. the spatial dependence is "flattened"  
    # initialize CS fields at zero as default 

    # CS fields 
    self.phi = np.zeros((Nx**dim, ntau), dtype=np.complex_)
    self.phistar = np.zeros((Nx**dim, ntau), dtype=np.complex_)
     
    # total forces for CS fields  
    self.dSdphi = np.zeros((Nx**dim, ntau), dtype=np.complex_)
    self.dSdphistar = np.zeros((Nx**dim, ntau), dtype=np.complex_)
    
    # Linear force coefficients 
    self.lincoef = np.zeros((Nx**dim, ntau), dtype=np.complex_)
    self.lincoef_phistar = np.zeros((Nx**dim, ntau), dtype=np.complex_)
    # Optional Linear force shifting coefficient
    self.Bn = np.zeros((Nx**dim, ntau), dtype=np.complex_)
    self.Bn_star = np.zeros((Nx**dim, ntau), dtype=np.complex_)
 
    # Initialize CS fields
    self.initialize_CSfields('constant') 
    #self.initialize_CSfields('random') 

    # Setup the k^2 grid, necessary for spectral evaluation of kinetic energy operator on CSfields 
    #   For single site model (Nx = 1), the reciprocol lattice "kgrid" is just the k = 0 point.  
    self.k2grid = None  
    self.setup_kgrid()

    # Get the linear force coefficients  
    self.fill_lincoefs()

    # Fill the nonlinear forces 
    self.fill_forces() 

    # Create operators 
    self.N_operator = None




  def initialize_CSfields(self, IC_type):
    IC = np.zeros((self.N_spatial, self.ntau), dtype=np.complex_)
    if(IC_type == 'constant'):
      N_input = 0.
      if(self.mu < 0 or self.g == 0):
        N_input = 25.0 # ideal gas usually is around 10 - 100 particles. No analytical approx necessary   
      else:
        N_input = (self.mu / self.g * self.Volume) # particle number input 
      IC += np.sqrt(N_input/self.Volume) 

    elif(IC_type == 'random'):
      # option to do normally distributed random nums
      IC += np.random.normal(0, 1.0, (self.N_spatial, self.ntau)) + 1j*np.random.normal(0., 1.0, (self.N_spatial, self.ntau)) 

    # else: leave as zeros 
    print('Initializing phi and phistar')
    self.phi += IC
    self.phistar += np.conj(IC)


  def setup_kgrid(self):
    print('Setting up the k^2 grid')
    # Set up the spatial/k-grids
    if(self.Nx > 3):
      n_grid = np.append( np.arange(0, self.Nx/2 + 1, 1) , np.arange(-self.Nx/2 + 1, 0, 1) ) # artifical +1 required for 2nd argument 
    elif( self.Nx == 1):
      n_grid = np.array([0]) 
    dk = np.pi * 2 / self.L
    
    x_grid = np.arange(0., self.L, self.L/self.Nx) 
    kx_grid = n_grid * dk
    assert(len(x_grid) == len(kx_grid))
    if(self.dim > 1):
      if(self.dim > 2):
        z_grid = x_grid 
        kz_grid = kx_grid 
      y_grid = x_grid # assumes cubic mesh  
      ky_grid = kx_grid 

    # these are not synced up with k-grid 
    if(self.dim > 1):
      X,Y = np.meshgrid(x_grid, y_grid)
      if(self.dim > 2):
        X,Y,Z = np.meshgrid(x_grid, y_grid, z_grid)
    else:
      X = x_grid
    
    _k2_grid = np.zeros(self.Nx**self.dim)
    _k2_grid_v2 = np.zeros(self.Nx**self.dim)
    if(self.dim > 1):
      KX, KY = np.meshgrid(kx_grid, ky_grid) 
      _k2_grid_v2 += (KX*KX + KY*KY).flatten()
      if(self.dim > 2):
        KX, KY, KZ = np.meshgrid(kx_grid, ky_grid, kz_grid) 
        _k2_grid_v2 += (KX*KX + KY*KY + KZ*KZ).flatten()
    else:
      KX = kx_grid 
      _k2_grid_v2 += (KX*KX).flatten() 
    _k2_grid += _k2_grid_v2
    # return/initialize_k2_grid 
    self.k2grid = _k2_grid 


  def fill_lincoefs(self):
    # Loop through Matsubara frequencies 
    for n in range(0, self.ntau):
      self.lincoef[:,n] =  A_nk(n, self.ntau, self.beta, self.k2grid, self._lambda, self.ensemble, self.mu)
      self.lincoef_phistar[:,n] = np.conj(self.lincoef[:,n])

#  def setup_operators(self, ops_list, N_samples):
#    #self.operators_list = ops_list
#    for op in ops_list:
#      Operator(
#      self.create_operator(op, N_samples) # allocate space for operator 

  def fill_forces(self):
    # Fills the nonlinear forces as applicable  
    self.dSdphistar.fill(0.)
    self.dSdphi.fill(0.)
    self.Bn.fill(0.) 
    self.Bn_star.fill(0.) 
  
    avg_rho = np.ceil( self.mu/self.g ) # a parameter for the stabilizing shift B(n) 
    #avg_rho = 1.5 
    #dtau = _beta/ntau
    for itau in range(0, int(self.ntau)):
      # PBC 
      itaum1 = ( (int(itau) - 1) % int(self.ntau) + int(self.ntau)) % int(self.ntau)
      # Nonlinear forces 
      # dSdphistar_j = g * phi^*_j phi_j-1 phi_j-1 
      # dSdphi_j = g * phi^*_j+1 phi_j phi_j 
      self.dSdphistar[:, itau] += self.g * self.dtau * self.phi[:, itaum1] * self.phi[:, itaum1] * self.phistar[:, itau]
      self.dSdphi[:, itaum1] += self.g * self.dtau * self.phistar[:, itau] * self.phi[:, itaum1] * self.phistar[:, itau]
      if(self.isShifting):
        self.Bn[:, itau] += avg_rho * self.dtau * self.g * np.exp(-2. * np.pi * 1j * itau / self.ntau) 
        # Other attempts: 
        #self.Bn[:, itau] = 0. 
        #self.Bn[:, itau] += itau         # 
        #self.Bn[:, itau] += itau*itau 
        # 1. shift to ensure pos. definite net linear coefficient 
        if((self.Bn[:, itau] + self.lincoef[:,itau]).real < 0.):
          self.Bn[:, itau] -= 2.*(self.Bn[:, itau] + self.lincoef[:, itau]).real # reflect onto real axis  
        #print( (self.Bn[:, itau] + self.lincoef[:,itau]).real >= 0.) 
  
        # 2. shift to ensure real linear coefficient 
        if(np.abs((self.Bn[:, itau] + self.lincoef[:,itau]).imag) > 0.):
          self.Bn[:, itau] -= 1j*(self.Bn[:, itau] + self.lincoef[:, itau]).imag
        #print( np.abs((self.Bn[:, itau] + self.lincoef[:,itau]).imag) < 1E-10) # check that we've eliminated the imaginary part  
        self.Bn_star[:, itau] = np.conj(self.Bn[:, itau]) 

    if(self.isShifting): 
      # FFT CS fields and forces 
      self.phi = fft_dp1(self.phi)
      self.phistar = fft_dp1(self.phistar)
      self.dSdphistar = fft_dp1(self.dSdphistar)
      self.dSdphi = fft_dp1(self.dSdphi)
  
      self.dSdphistar -= (self.Bn * self.phi) # must be in matsubara representation
      self.dSdphi -= (self.Bn_star * self.phistar)
      # iFFT back; rest of the code expects forces and phi in r, tau space 
      self.phi = ifft_dp1(self.phi)
      self.phistar = ifft_dp1(self.phistar)
      self.dSdphistar = ifft_dp1(self.dSdphistar)
      self.dSdphi = ifft_dp1(self.dSdphi)



## Helper function for calculating Linear force coefficients 
def A_nk(n, _ntau, _beta, k2_grid, _lambda, _ensemble, _mu):
        # Returns a d-dimensional field, evaluated at timeslice "n" 
        A = np.zeros(len(k2_grid), dtype=np.complex_) 
        dtau = _beta/_ntau
        A += (- dtau * _lambda * k2_grid) # hbar^2/2m k^2, kinetic energy 
        if(_ensemble == "GRAND"):
          A += (_mu * dtau)
        A += 1. 
        A *= -np.exp(-2. * np.pi * 1j * n / _ntau) 
        A += 1.
        return A 

