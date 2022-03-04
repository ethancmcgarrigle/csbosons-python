import numpy as np
from numpy import linalg
import yaml
import math
import matplotlib
# matplotlib.rcParams['text.usetex'] = True
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt 
import time
from scipy.fft import fft 
from scipy.fft import ifft
import matplotlib 
import matplotlib.pyplot as plt 
import copy
import io_functions

# Toy model
suppressOutput = True



def cotangent(x):
  _result = np.tan(x)
  _result = 1./_result # inverse of tangent is cotangent 
  return _result


# Derivatives 
def dS_dphi(phi_vec, beta1, beta2):
  y = copy.deepcopy(phi_vec) 
  derivatives = np.zeros(len(y), dtype=np.complex_) # should be 3 
 
  for i in range(0, len(y)): 
    derivatives[i] = -beta1 * 1j * (np.exp(1j * y[i]) - np.exp(-1j * (y[0] + y[1])) ) 
    derivatives[i] += 1j * beta2 * ( np.exp(-1j * y[i]) - np.exp(1j * (y[0] + y[1])) )
    if(i == 0):
      derivatives[i] -= cotangent(0.5 * (y[0] - y[1]))
      derivatives[i] -= cotangent(0.5 * (y[0] + 2. * y[1]))
      derivatives[i] -= 2. * cotangent(0.5 * (2. * y[0] + y[1]))
    elif(i == 1): 
      derivatives[i] += cotangent(0.5 * (y[0] - y[1]))
      derivatives[i] -= cotangent(0.5 * (2. * y[0] + y[1]))
      derivatives[i] -= 2. * cotangent(0.5 * (y[0] + 2. * y[1]))

  return derivatives 


# Traces
def tr_U(phi_vec, inv_bool, exponent = 1.):
  _result = 0. + 1j*0.
  y = copy.deepcopy(phi_vec) 
  y = np.append(y, -(y[0] + y[1]))
  if(inv_bool):
    y *= -1j
  else: 
    y *= 1j
  
  y = np.exp(y) # e^{i phi}
  y **= exponent 
  _result = np.sum(y)
  return _result



def tr_U_Udag(phi_vec):
  _result = 0. + 1j*0.
  y = copy.deepcopy(phi_vec) 
  y = np.append(y, -(y[0] + y[1]))
  y *= 1j
  y2 = np.zeros(len(y), dtype = np.complex_) # i phi
  y2 -= y # -i phi
  y = np.exp(y) # e^{i phi} 
  y += np.exp(y2) # e{i phi} + e^{-i phi} 
  # take the trace 
  _result = np.sum(y) * 0.5
  return _result


def density(phi_vec, h, mu):
  _result = 0. + 1j*0.
  y = copy.deepcopy(phi_vec) # y is appended inside of the Trace function 
  _result += tr_U(y, False, 1.) * h * np.exp(mu)
  _result -= h * np.exp(-1. * mu) * tr_U(y, True, 1.)
  return _result




# Evaluate constraint 
def constraint(phi_vec):
  _result = 0. + 1j*0.
  y = copy.deepcopy(phi_vec) 
  y = np.append(y, -(y[0] + y[1]))
  _result = np.sum(y) # should be satisfied automatically
  return _result




# Load the inputs 
parameters = io_functions.open_params('params.yml') 
_beta1, _beta2, _mu, _h, dt, numtsteps, iofreq, seed, isPlotting = parameters
np.random.seed(seed)

# Set up the simulation 

num_points = math.floor(numtsteps/iofreq)
[t_s, Tr_U_s, Tr_Usq_s, Tr_U_Udag_s, constraint_s, rho_s] = io_functions.initialize_Observables(num_points)



# initialize phi_vector 
phi_vector = np.zeros(2, dtype = np.complex_)
phi_vector[0] = 0.14
phi_vector[1] = 0.32


mobility = np.ones(len(phi_vector))
noisescls = np.zeros(len(phi_vector))
noisescls += np.sqrt(2. * mobility * dt) 


# Initialize with IC  
Tr_U_s[0] = tr_U(phi_vector, False) 
Tr_U_Udag_s[0] = tr_U_Udag(phi_vector) 
Tr_Usq_s[0] = tr_U(phi_vector, False, 2) 
constraint_s[0] = constraint(phi_vector)
rho_s[0] = density(phi_vector, _h, _mu)

Tr_U_avg = Tr_U_s[0]
Tr_U_Udag_avg = Tr_U_Udag_s[0]
Tr_Usq_avg = Tr_Usq_s[0]
constraint_avg = constraint_s[0]
rho_avg = rho_s[0]


# Begin sampling
ctr = 1
start = time.time()
t = 0.

observables = [constraint_avg, Tr_U_avg, Tr_Usq_avg, Tr_U_Udag_avg, rho_avg]
io_functions.write_observables("data_No_psi.dat", observables, t, True)
# Timestep using EM 
for l in range(0, numtsteps + 1):

  # Step each field using Euler Maruayama 
  phi_vector -= mobility * dt * (dS_dphi(phi_vector, _beta1, _beta2)) 
  
  # add noise 
  for i in range(0, len(phi_vector)):
    eta = np.random.normal() * noisescls[i]
    phi_vector[i] += eta 


  # Do block averaging 
  Tr_U_avg += tr_U(phi_vector, False)/iofreq
  Tr_Usq_avg += tr_U(phi_vector, False, 2.)/iofreq
  Tr_U_Udag_avg += tr_U_Udag(phi_vector)/iofreq
  constraint_avg += constraint(phi_vector)/iofreq
  rho_avg += density(phi_vector, _h, _mu)/iofreq
    
  t += dt

  # Output on interval
  if(l % iofreq == 0 and l > 0):
    # if(ctr %  25):

    if(not(suppressOutput)):
      print("Completed {} of {} steps".format(l, numtsteps))
    
    # opout.write("{} {} {} {} {}\n".format(it, Msum.real / Navg, Msum.imag / Navg, psi.real, psi.imag))
    observables = [constraint_avg, Tr_U_avg, Tr_Usq_avg, Tr_U_Udag_avg, rho_avg]
    io_functions.write_observables("data_No_psi.dat", observables, t, False)
    t_s[ctr] = t 
    if(math.isnan(constraint_avg.real)):
      print("Trajectory diverged -- nan, ending simulation")
      break
    Tr_U_s[ctr] = Tr_U_avg 
    Tr_U_Udag_s[ctr] = Tr_U_Udag_avg 
    Tr_Usq_s[ctr] = Tr_Usq_avg 
    constraint_s[ctr] = constraint_avg
    rho_s[ctr] = rho_avg 
    # clear the averages 
    Tr_U_avg = 0. + 1j*0 
    Tr_U_Udag_avg = 0. + 1j*0 
    Tr_Usq_avg = 0. + 1j*0 
    constraint_avg = 0. + 1j*0 
    rho_avg = 0. + 1j*0 
    ctr += 1

    

end = time.time()


# Package observables in a list 
observables = [constraint_s, Tr_U_s, Tr_Usq_s, Tr_U_Udag_s, rho_s]
if(not(suppressOutput)):
  io_functions.print_sim_output(t, observables)

# Prepare operator output stream
# io_functions.write_observables("data_No_psi.dat", observables, t) 
# io_functions.write_operators("operators_no_psi.dat", observables, t_s)

# plots 
if(isPlotting):
  io_functions.plot_CL_traces(t_s, observables)


