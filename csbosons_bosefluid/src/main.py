import numpy as np
import yaml
import math
import time
from scipy.fft import fft 
from scipy.fft import ifft
import matplotlib 
import matplotlib.pyplot as plt 
from scipy.stats import sem
# Import our custom classes 
from Bosefluid_Model import Bosefluid_Model
from CL_Driver import CL_Driver
import Operator

###################### Input Parameters ########################
''' Python code for running Coherent States boson models. 
    Inputs: 
           mu:  chemical potential
            g:  pairwise interactiong strength (density-density interaction)
         beta:  inverse temperature (1/T)
          dim:  spatial dimension 
            L:  size of the simulation cell, assuming square/cubic geometries with periodic boundary conditions
           Nx:  number of grid points in each direction
           dt:  Langevin time step 
   numtisteps:  number of timesteps to run in total
       iofreq:  frequency of printing samples. We block average observables 
     ensemble:  GRAND for grand canonical, CANONICAL (NVT) is disabled in this code 

    Outputs: 
        Particle number:   The total number of boson particles in the system 

 - We expect mean-field solutions where phi = phistar = \sqrt(mu / g); valid when repulsive energy scale (g) is small. 
 - This corresponds to N = mu/g for a single site model (L = Nx = 1). 
 - In the strongly repulsive regime at low temperatures, we expect N to become integer valued, where it equals np.ceil(mu/g).  
     - We have not had success simulating in the strongly repulsive regime. ''' 

########### System ############### 
ensemble = 'GRAND'  # Ensemble. either grand or canonical
_mu = 1.0    # Chemical potential 
_g = 0.10 #  Repulsion interaction strength     # ideal gas if g == 0 
_beta = 1.00 # inverse temperature 
_lambda = 6.0505834240  # hbar^2 / 2m for mass of He4; not necessary for single-site lattice model 
dim = 1  # testing in 1D
L = 1  # simulation box size. L = 1 with Nx = 1 (1 grid point) corresponds to a single-site Bose-Hubbard (lattice) model 
ntau = 32     # number of imaginary time points  
Vol = L**dim


######## Simulation, Discretization parameters ########### 
Nx = 1  # number of grid points; assumes cubic/square cells (Nx = Ny = Nz) 
dt = 0.01  # Langevin timestep 
numtsteps = 10000 # total number of Langevin steps 
iofreq = 100   # print every ___ Langevin steps  
num_samples = math.floor(numtsteps/iofreq)
_CLnoise = True  # Complex Langevin (with Noise or true) or mean-field theory (no noise / false) 
_ETD = True   # Exponential time differencing if true; Euler Maruyama if false 
_isShifting = False #Boolean for shifting linear coefficients for stability testing 

assert L == 1 and Nx == 1, 'This code is meant to test the single site model, corresponding to L = 1 and Nx = 1'
#_do_implicit = True   # Fully implicit iteraton currently doesn't work in this formulation of the code 

np.random.seed(1)  # Use a consistent pseudorandom number seed for testing 

# Create Coherent-State Field Model, initialize CSfields and spatial/tau grids, and all forces and operators  
_boson_model = Bosefluid_Model(_mu, _g, _beta, _lambda, dim, L, ntau, ensemble, Nx, _isShifting) 

# Create the Driver  
_simulation_driver = CL_Driver(_boson_model, _CLnoise, _ETD, _isShifting, dt, numtsteps, iofreq, num_samples)


print('Initial particle number: ' + str(_boson_model.N_operator.returnParticleNumber()) )
print()
print('initial (average) density : ' + str(_boson_model.N_operator.returnParticleNumber()/Vol))

# Start the simulation 
_isPlotting = True
_simulation_driver.run_simulation(_isPlotting)

print('Code exited successfully')


