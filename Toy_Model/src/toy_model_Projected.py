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
import sys


start = time.time()


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
    derivatives[i] = -beta1 * 1j * np.exp(1j * y[i]) 
    derivatives[i] += 1j * beta2 * np.exp(-1j * y[i])
    if(i == 0):
      derivatives[i] -= cotangent(0.5 * (y[0] - y[1]))
      derivatives[i] -= cotangent(0.5 * (y[0] - y[2]))
    elif(i == 1):
      derivatives[i] += cotangent(0.5 * (y[0] - y[1]))
      derivatives[i] -= cotangent(0.5 * (y[1] - y[2]))
    elif(i == 2): 
      derivatives[i] += cotangent(0.5 * (y[0] - y[2]))
      derivatives[i] += cotangent(0.5 * (y[1] - y[2]))

  return derivatives 


# Traces
def tr_U(phi_vec, inv_bool, exponent = 1.):
  y = copy.deepcopy(phi_vec) 
  _result = 0. + 1j*0.
  if(inv_bool):
    y *= -1j
  else: 
    y *= 1j
  
  y = np.exp(y) # e^{i phi}
  y **= exponent
  _result = np.sum(y) # Trace 
  return _result

def tr_U_Udag(phi_vec):
  _result = 0. + 1j*0.
  y = copy.deepcopy(phi_vec) 
  y *= 1j
  tmp = np.zeros(len(y), dtype=np.complex_)
  tmp -= y # i phi
  y = np.exp(y) # e^{i phi} 
  y += np.exp(tmp) # e{i phi} + e^{-i phi} 
  # take the trace 
  _result = np.sum(y) * 0.5
  return _result


def density(phi_vec, h, mu):
  _result = 0. + 1j*0.
  y = copy.deepcopy(phi_vec)
  _result += tr_U(y, False, 1.) * h * np.exp(mu)
  _result -= h * np.exp(-1. * mu) * tr_U(y, True, 1.)
  return _result



# Evaluate constraint 
def constraint(phi_vec):
  _result = 0. + 1j*0.
  _result = np.sum(phi_vec)
  return _result


# Load the inputs
inputfile = str(sys.argv[1]) 
parameters = io_functions.open_params(inputfile) 
_beta1, _beta2, _mu, _h, dt, numtsteps, iofreq, seed, isPlotting = parameters
np.random.seed(seed)

# Set up the simulation 

num_points = math.floor(numtsteps/iofreq)
[t_s, Tr_U_s, Tr_Usq_s, Tr_U_Udag_s, constraint_s, rho_s] = io_functions.initialize_Observables(num_points)

# initialize phi_vector 
phi_vector = np.zeros(3, dtype = np.complex_)
phi_vector[0] = 1.57
phi_vector[1] = 1.57/2
phi_vector[2] = -1.5 * 1.57

# initialize psi field (lagrange multiplier)
_psi = 0. + 1j * 0. 


# Initialize with IC 
psi_s = np.zeros(num_points + 1, dtype=np.complex_)
num_iters_s = np.zeros(num_points + 1)
  
psi_s[0] = _psi
Tr_U_s[0] = tr_U(phi_vector, False) 
Tr_Usq_s[0] = tr_U(phi_vector, False, 2) 
Tr_U_Udag_s[0] = tr_U_Udag(phi_vector) 
constraint_s[0] = constraint(phi_vector)
rho_s[0] = density(phi_vector, _h, _mu)
num_iters_s[0] = 1

Tr_U_avg = Tr_U_s[0]
Tr_Usq_avg = Tr_Usq_s[0]
Tr_U_Udag_avg = Tr_U_Udag_s[0]
constraint_avg = constraint_s[0]
rho_avg = rho_s[0]
num_iters_avg = num_iters_s[0]

# set mobilities and noise scalings
mobility = 0.1 * np.ones(len(phi_vector))
_psi_mobility = 0.5
noisescls = np.zeros(len(phi_vector))
noisescls += np.sqrt(2. * mobility * dt) 


# Begin sampling
ctr = 1
t = 0.

# Initialize Matrices
ntau = 1 
G_matrix = np.zeros((4*ntau, 4*ntau), dtype=np.complex_) 
d_vector = np.zeros(4*ntau, dtype=np.complex_) 


# Fill G_matrix (only needs to be done once because it's a linear problem)
for i in range(0, 3):
  G_matrix[i][i] = 1.
  G_matrix[i][3] = -2. 
  G_matrix[3][i] = 1.  

# grad e
grad_e = np.ones(len(phi_vector), dtype=np.complex_)
phi_vector_iter = np.zeros(len(phi_vector), dtype=np.complex_) 
y_tilde = np.zeros(len(phi_vector), dtype=np.complex_) 
y1 = np.zeros(len(phi_vector), dtype=np.complex_)

ETOL = 1E-6
max_iterations = 250


observables = [constraint_avg, Tr_U_avg, Tr_Usq_avg, Tr_U_Udag_avg, rho_avg]
io_functions.write_observables("data_projected.dat", observables, t, True)

opout=open('operators_psi_proj.dat',"w")
opout.write("# t_elapsed psi.real psi.imag avg_iters_per_dt \n")
opout.write("{} {} {} {}\n".format(t, _psi.real, _psi.imag, num_iters_s[0]))
opout.close() 

# Timestep using EM 
for l in range(0, numtsteps + 1):

  # Generate noise and scale 
  noises = np.random.normal(0, 1., len(phi_vector)) # real noise 
  noises *= noisescls 
 
  # Set up Newton's iteration
  # Use previous psi as initial iterate guess for psi  
  psi_iter = 0. + 1j * 0.
  psi_iter += _psi
  # initial iterate/guess: phi_vector EM stepped with psi term (typical CL)
  phi_vector_iter.fill(0.) # reset to zero 
  phi_vector_iter += phi_vector
  phi_vector_iter -= mobility * dt * (dS_dphi(phi_vector, _beta1, _beta2) + 1j * _psi)
  num_iters = 0

  for i in range(0, max_iterations + 1):
    # Do the Naive Euler step
    # calculate y_tilde 
    y_tilde.fill(0.)
    y_tilde += phi_vector
    y_tilde += grad_e * psi_iter  # current psi iteration 
    # Do Euler maruyama on y_tilde
    y_EM = y_tilde - mobility * dt * (dS_dphi(y_tilde, _beta1, _beta2)) + noises
    
    # reset size of y1 vector back to 3 entries
    # fill it with the next iteration  
    y1.fill(0.) # zero 
    y1 += phi_vector_iter  
    
    d_vector[3] = constraint(y1) 
    y1 -= (grad_e * psi_iter)
    # Take EM step 
    # phi_vector -= mobility * dt * (dS_dphi(phi_vector, _beta1, _beta2))
    y1 -= y_EM # residuals 
   
    # y1 = y1.append(y1, e_y1)
    d_vector[0:3] = y1
    
    d_vector *= -1. # "b" vector to do Ax = b
    # do Ax = b, using G as A and delta as b 
    solution = np.linalg.solve(G_matrix, d_vector)
    
    # Update iteration with the solution (updates) 
    phi_vector_iter += solution[0:3]  
    psi_iter += solution[3]
    
    if(np.all(solution < ETOL)):
      num_iters += i
      break;  

  if( i == max_iterations):
    print('Warning: Max iterations exceeded')
    num_iters += max_iterations


  num_iters += 1
  # Step each field using Euler Maruayama 
  phi_vector.fill(0.)
  _psi = 0. + 1j * 0.
  _psi += psi_iter
  phi_vector += phi_vector_iter
  
  # Do block averaging 
  Tr_U_avg += tr_U(phi_vector, False)/iofreq
  Tr_Usq_avg += tr_U(phi_vector, False, 2.)/iofreq
  Tr_U_Udag_avg += tr_U_Udag(phi_vector)/iofreq
  constraint_avg += constraint(phi_vector)/iofreq
  rho_avg += density(phi_vector, _h, _mu)/iofreq
  num_iters_avg += num_iters/iofreq 
 
  t += dt

  # Output on interval
  if(l % iofreq == 0 and l > 0):
    # if(ctr %  25):

    if(not(suppressOutput)):
      print("Completed {} of {} steps".format(l, numtsteps))
    # opout.write("{} {} {} {} {}\n".format(it, Msum.real / Navg, Msum.imag / Navg, psi.real, psi.imag))
    t_s[ctr] = t 
    if(math.isnan(constraint_avg.real)):
      print("Trajectory diverged -- nan, ending simulation")
      break

    observables = [constraint_avg, Tr_U_avg, Tr_Usq_avg, Tr_U_Udag_avg, rho_avg]
    io_functions.write_observables("data_projected.dat", observables, t, False)
    num_iters_s[ctr] = num_iters_avg 

    opout=open('operators_psi_proj.dat',"a")
    opout.write("{} {} {} {}\n".format(t, _psi.real, _psi.imag, num_iters_avg))
    opout.close() 

    psi_s[ctr] = _psi
    Tr_U_s[ctr] = Tr_U_avg 
    Tr_Usq_s[ctr] = Tr_Usq_avg 
    Tr_U_Udag_s[ctr] = Tr_U_Udag_avg 
    constraint_s[ctr] = constraint_avg
    rho_s[ctr] = rho_avg 
    # clear the averages 
    Tr_U_avg = 0. + 1j*0 
    Tr_U_Udag_avg = 0. + 1j*0 
    Tr_Usq_avg = 0. + 1j*0 
    constraint_avg = 0. + 1j*0 
    rho_avg = 0. + 1j*0 
    ctr += 1
    num_iters_avg = 0.




print('Elapsed Time : ' + str(time.time() - start) + ' seconds \n')

# Package observables in a list 
observables = [constraint_s, Tr_U_s, Tr_Usq_s, Tr_U_Udag_s, rho_s]
if(not(suppressOutput)):
  io_functions.print_sim_output(t, observables)

# Prepare operator output stream
# io_functions.write_observables("data_projected.dat", observables, t)

# plots 
if(isPlotting):
  io_functions.plot_CL_traces(t_s, observables)


if(isPlotting):
  plt.figure(13)
  plt.title('Iterations per timestep sampling: CL Simulation', fontsize = 20, fontweight = 'bold')
  plt.plot(t_s, num_iters_s, '-r', label = 'Samples')
  plt.xlabel('CL time', fontsize = 20, fontweight = 'bold')
  plt.ylabel('Iterations', fontsize = 20, fontweight = 'bold') 
  # plt.ylim([-5, 5])
  plt.legend()
  plt.show()
  
  plt.figure(2)
  plt.title('Psi sampling: CL Simulation', fontsize = 20, fontweight = 'bold')
  plt.plot(t_s, psi_s.real, '-r', label = 'Samples: real')
  plt.plot(t_s, psi_s.imag, '-g', label = 'Samples: imag')
  plt.xlabel('CL time', fontsize = 20, fontweight = 'bold')
  plt.ylabel('$\psi$', fontsize = 20, fontweight = 'bold') 
  # plt.ylim([-5, 5])
  plt.legend()
  plt.show()
  
  plt.figure(3)
  plt.title('Psi Trajectory: CL Simulation', fontsize = 20, fontweight = 'bold')
  plt.plot(psi_s.real, psi_s.imag, '-r', label = 'Samples: real')
  plt.xlabel('Re($\psi$)', fontsize = 20, fontweight = 'bold')
  plt.ylabel('Im($\psi$)', fontsize = 20, fontweight = 'bold') 
  plt.legend()
  # plt.ylim([-2, 1])
  plt.show()
