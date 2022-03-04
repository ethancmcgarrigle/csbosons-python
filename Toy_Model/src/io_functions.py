import numpy as np
from numpy import linalg
import yaml
import math
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt 
import time
from scipy.fft import fft 
from scipy.fft import ifft
import matplotlib 
import matplotlib.pyplot as plt 
import copy

# Input/output functions for the toy models



def open_params(filename):
  # filename is a string, should be .yml extension 
  # Read simulation parameters
  with open(filename) as infile:
      params = yaml.load(infile, Loader=yaml.FullLoader)

  # Store parameters in local variables and echo to screen
  _beta = params['model']['beta']
  _mu = params['model']['mu']
  _gamma = params['model']['gamma'] # phase of u
  _u = 1./(3. * _beta) # b_eff = 2 
  # _u = 1./(12. * _beta) # b_eff = 0.5 => 6 _beta u => u = (1/12)*(1/beta) 
  _h = params['model']['h'] 
  
  _u *= np.exp(1j * _gamma)
  
  dt = float(params['timestepping']['dt'])
  numtsteps = params['timestepping']['ntmax']
  iofreq = params['timestepping']['io_interval']
  seed = params['timestepping']['seed']
  isPlotting = params['timestepping']['isPlotting']
  _beta1 = 6. * _beta * _u
  _beta2 = np.conj(_beta1)
  
  _beta1 += (_h * np.exp(_mu))
  _beta2 += (_h * np.exp(-1.*_mu))
  parameters = [_beta1, _beta2, _mu, _h, dt, numtsteps, iofreq, seed, isPlotting]
  print()
  print()
  print('----- SU(3) Simulation, 1-link effective model (no x dependence) ----')
  print()
  print()
  print('dt : ' + str(dt))
  print()
  print('beta_1 : ' + str(_beta1))
  print()
  print('beta_2 : ' + str(_beta2))
  print()
  print('h: ' + str(_h))
  print()
  print('mu: ' + str(_mu))
  print()
  print('Running for ' + str(numtsteps) + ' timesteps')
  print()
  print()
  print('Printing every ' + str(iofreq) + ' timesteps')
  print()
  print('Random Number Seed ' + str(seed))
  print()
  print('Is plotting? ' + str(isPlotting))
  print()
  print()
  print('Complex Langevin Sampling')
  print()
  return parameters 



def initialize_Observables(num_points):
  # returns list of observables 
  # Sampling vectors
  t_s = np.zeros(num_points + 1)
  Tr_U_s = np.zeros(num_points + 1, dtype=np.complex_)
  Tr_Usq_s = np.zeros(num_points + 1, dtype=np.complex_)
  Tr_U_Udag_s = np.zeros(num_points + 1, dtype=np.complex_)
  constraint_s = np.zeros(num_points + 1, dtype=np.complex_)
  rho_s = np.zeros(num_points + 1, dtype=np.complex_)
  global start 
  start = time.time()
  
  return [t_s, Tr_U_s, Tr_Usq_s, Tr_U_Udag_s, constraint_s, rho_s]




def write_observables(filename, observables, t, isFirstLine):
  if(isFirstLine):
    opout=open(filename,"w")
  else:
    opout=open(filename,"a")

  # Observables is a list of complex numbers, block averages from CL 
  constraint_s, Tr_U_s, Tr_Usq_s, Tr_U_Udag_s, rho_s = observables
  if(isFirstLine):
    opout.write("# t_elapsed Tr_U.real Tr_U.imag Tr_U2.real Tr_U2.imag Tr_UUdag.real Tr_UUdag.imag constraint.real constraint.imag rho.real rho.imag \n")

  opout.write("{} {} {} {} {} {} {} {} {} {} {}\n".format(t, Tr_U_s.real, Tr_U_s.imag, Tr_Usq_s.real, Tr_Usq_s.imag, Tr_U_Udag_s.real, Tr_U_Udag_s.imag, constraint_s.real, constraint_s.imag, rho_s.real, rho_s.imag ))
  # opout.write("{} {} {} {} {} {} {} {} {} {} {}\n".format(t, np.mean(Tr_U_s[100:]).real, np.mean(Tr_U_s[100:]).imag, np.mean(Tr_Usq_s[100:]).real, np.mean(Tr_Usq_s[100:]).imag, np.mean(Tr_U_Udag_s[100:]).real, np.mean(Tr_U_Udag_s[100:]).imag, np.mean(constraint_s[100:]).real, np.mean(constraint_s[100:]).imag, np.mean(rho_s[100:]).real, np.mean(rho_s[100:]).imag ))
  opout.close()


 #def write_operators(filename, observables, t):
 #  np.savetxt(filename, np.column_stack([t, observables])) # , fmt = ['%.5f', '%.5f', '%.5f', '%.5f', '%.5f', '%.5f', '%.5f'])



def print_sim_output(t, observables):
  # Observables is a list of numpy arrays -- representing column simulation data 
  end = time.time()
  constraint_s, Tr_U_s, Tr_Usq_s, Tr_U_Udag_s, rho_s = observables
  print()
  print()
  print('Simulation finished: Runtime = ' + str(end - start) + ' seconds')
  print()
  print('Printing Thermodynamic Averages')
  print()
  print()
  print('The Constraint is: ' + str(np.mean(constraint_s[100:])))
  print()
  print('Trace of U : ' + str(np.mean(Tr_U_s[100:])))
  print()
  print('Trace of U squared : ' + str(np.mean(Tr_Usq_s[100:])))
  print()
  print('1/2 of Trace of U + U-dagger: ' + str(np.mean(Tr_U_Udag_s[100:])))
  print()
  print('The density is: ' + str(np.mean(rho_s[100:])))
  print()
  print('Writing results to a .dat file')
  

def plot_CL_traces(t_s, observables):
  constraint_s, Tr_U_s, Tr_Usq_s, Tr_U_Udag_s, rho_s = observables
  plt.figure(1)
  plt.title('Tr(U) sampling: CL Simulation', fontsize = 20, fontweight = 'bold')
  plt.plot(t_s, Tr_U_s.real, '-r', label = 'Samples: real')
  plt.plot(t_s, Tr_U_s.imag, '-g', label = 'Samples: imag')
  plt.xlabel('CL time', fontsize = 20, fontweight = 'bold')
  plt.ylabel('Tr[$U$]', fontsize = 20, fontweight = 'bold') 
  plt.ylim([-15, 15])
  plt.legend()
  plt.show()
  
  plt.figure(6)
  plt.title('Tr$(U^2)$ sampling: CL Simulation', fontsize = 20, fontweight = 'bold')
  plt.plot(t_s, Tr_Usq_s.real, '-r', label = 'Samples: real')
  plt.plot(t_s, Tr_Usq_s.imag, '-g', label = 'Samples: imag')
  plt.xlabel('CL time', fontsize = 20, fontweight = 'bold')
  plt.ylabel('Tr[$U^2$]', fontsize = 20, fontweight = 'bold') 
  plt.ylim([-15, 15])
  plt.legend()
  plt.show()
  
  plt.figure(4)
  plt.title('$1/2 (Tr(U + U^{-1}))$ sampling: CL Simulation', fontsize = 20, fontweight = 'bold')
  plt.plot(t_s, Tr_U_Udag_s.real, '-r', label = 'Samples: real')
  plt.plot(t_s, Tr_U_Udag_s.imag, '-g', label = 'Samples: imag')
  plt.xlabel('CL time', fontsize = 20, fontweight = 'bold')
  plt.ylabel('Tr[$U + U^{-1}$]', fontsize = 20, fontweight = 'bold') 
  plt.ylim([-15, 15])
  plt.legend()
  plt.show()
  
  plt.figure(7)
  plt.title('Density sampling: CL Simulation', fontsize = 20, fontweight = 'bold')
  plt.plot(t_s, rho_s.real, '-r', label = 'Samples: real')
  plt.plot(t_s, rho_s.imag, '-g', label = 'Samples: imag')
  plt.xlabel('CL time', fontsize = 20, fontweight = 'bold')
  plt.ylabel('$n$', fontsize = 20, fontweight = 'bold') 
  plt.ylim([-15, 15])
  plt.legend()
  plt.show()
  
  
  plt.figure(5)
  plt.title('Constraint : CL Simulation', fontsize = 20, fontweight = 'bold')
  plt.plot(t_s, constraint_s.real, '-r', label = 'Samples: real')
  plt.plot(t_s, constraint_s.imag, '-g', label = 'Samples: imag')
  plt.xlabel('CL time', fontsize = 20, fontweight = 'bold')
  plt.ylabel('e', fontsize = 20, fontweight = 'bold') 
  #plt.ylim([-5, 5])
  plt.legend()
  plt.show()
