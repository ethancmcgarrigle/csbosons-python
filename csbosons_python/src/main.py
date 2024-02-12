import numpy as np
import yaml
import math
import matplotlib
import matplotlib.pyplot as plt 
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

########### System ############### 
ensemble = 'GRAND'  # Ensemble. either grand or canonical
_mu = 1.0    # Chemical potential 
_g = 0.10 # Interaction strength     # ideal gas if == 0 
_beta = 1.00 # inverse temperature 
_lambda = 6.0505834240  # hbar^2 / 2m for mass of He4
dim = 1
L = 1  # simulation box size 
ntau = 24     # number of imaginary time points 
Vol = L**dim

######## Simulation, Discretization parameters ########### 
Nx = 1  # assumes cubic/square cells (Nx = Ny = Nz) 
dt = 0.01      # Langevin timestep 
numtsteps = 10000 # total number of Langevin steps 
iofreq = 200 # print every ___ Langevin steps  
num_samples = math.floor(numtsteps/iofreq)
_CLnoise = True  # Complex Langevin (with Noise) or mean-field theory (no noise) 
_isOffDiagonal = True
_ETD = True
_do_implicit = False
_isShifting = True

np.random.seed(1)  # Use a consistent pseudorandom number seed for testing 

# Create Coherent-State Field Model, initialize CSfields and spatial/tau grids, and all forces and operators  
_boson_model = Bosefluid_Model(_mu, _g, _beta, _lambda, dim, L, ntau, ensemble, Nx, _isShifting) 

# Create the Driver  
_simulation_driver = CL_Driver(_boson_model, _CLnoise, _isOffDiagonal, _ETD, _do_implicit, _isShifting, dt, numtsteps, iofreq, num_samples)


print('Initial particle number: ' + str(_boson_model.N_operator.returnParticleNumber()) )
print()
print('initial (average) density : ' + str(_boson_model.N_operator.returnParticleNumber()/Vol))

# Start the simulation 
_isPlotting = True
_simulation_driver.run_simulation(_isPlotting)

print('Code exited successfully')

