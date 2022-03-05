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
  ntau = params['model']['ntau'] # phase of u
  _hz = params['model']['hz'] 
  _U = params['model']['U'] 
  
  dt = float(params['timestepping']['dt'])
  numtsteps = params['timestepping']['ntmax']
  iofreq = params['timestepping']['io_interval']
  seed = params['timestepping']['seed']
  isPlotting = params['timestepping']['isPlotting']

  parameters = [_beta, ntau, _hz, dt, numtsteps, iofreq, seed, isPlotting]
  print()
  print()
  print('----- Schwinger Bosonic Coherent States Simulation, 1 spin model  ----')
  print()
  print()
  print('dt : ' + str(dt))
  print()
  print('beta: ' + str(_beta))
  print()
  print('Temperature: ' + str(1./_beta))
  print()
  print('ntau: ' + str(ntau))
  print()
  print('hz: ' + str(_hz))
  print()
  print('U penalty strength: ' + str(_U))
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
  N_tot_s = np.zeros(num_points + 1, dtype=np.complex_)
  N_up_s = np.zeros(num_points + 1, dtype=np.complex_)
  N_dwn_s = np.zeros(num_points + 1, dtype=np.complex_)
  Mag_s = np.zeros(num_points + 1, dtype=np.complex_)
  M2_s = np.zeros(num_points + 1, dtype=np.complex_)
  psi_s = np.zeros(num_points + 1, dtype=np.complex_)
  global start 
  start = time.time()
  
  return [t_s, N_tot_s, N_up_s, N_dwn_s, psi_s, Mag_s, M2_s] 




def write_observables(filename, observables, t, isFirstLine):
  if(isFirstLine):
    opout=open(filename,"w")
  else:
    opout=open(filename,"a")

  # Observables is a list of complex numbers, block averages from CL 
  N_tot_s, N_up_s, N_dwn_s, psi_s, Mag_s, M2_s = observables
  if(isFirstLine):
    opout.write("# t_elapsed N_tot.real N_tot.imag N_up.real N_up.imag N_dwn.real N_dwn.imag Mag.real Mag.imag M2.real M2.imag \n")

  opout.write("{} {} {} {} {} {} {} {} {} {} {}\n".format(t, N_tot_s.real, N_tot_s.imag, N_up_s.real, N_up_s.imag, N_dwn_s.real, N_dwn_s.imag, Mag_s.real, Mag_s.imag, M2_s.real, M2_s.imag ))
  opout.close()



def print_sim_output(t, observables):
  # Observables is a list of numpy arrays -- representing column simulation data 
  end = time.time()
  N_tot_s, N_up_s, N_dwn_s, psi_s, Mag_s, M2_s = observables
  # Print the results (noise long-time averages)
  print()
  print('Simulation finished: Runtime = ' + str(end - start) + ' seconds')
  print()
  print('The Particle Number is: ' + str(np.mean(N_tot_s.real)))
  print()
  print('The Up Boson Particle Number is: ' + str(np.mean(N_up_s.real)))
  print()
  print('The Down Boson Particle Number is: ' + str(np.mean(N_dwn_s.real)))
  print()
  print('The Magnetization is: ' + str(np.mean(Mag_s.real)))
  print()
  print('The Magnetization-squared is: ' + str(np.mean(M2_s.real)))
  print()
  print('Writing results to a .dat file')
  print()
  

def plot_CL_traces(t_s, observables):
  N_tot_s, N_up_s, N_dwn_s, psi_s, Mag_s, M2_s = observables

  # plot the results 
  plt.figure(1)
  plt.title('Particle Number: CL Simulation', fontsize = 20, fontweight = 'bold')
  plt.plot(t_s, N_tot_s.real, '-', color = 'green', label = 'Samples: real')
  plt.plot(t_s, N_tot_s.imag, '-', color = 'skyblue', label = 'Samples: imag')
  plt.plot(t_s, np.ones(len(t_s)), 'k', label = 'Constraint')
  plt.xlabel('CL time', fontsize = 20, fontweight = 'bold')
  plt.ylabel('$N_{tot}$', fontsize = 20, fontweight = 'bold') 
  plt.ylim([-5, 10])
  plt.legend()
  plt.show()
  
  
  plt.figure(2)
  plt.title('Psi sampling: CL Simulation', fontsize = 20, fontweight = 'bold')
  plt.plot(t_s, psi_s.real, '-r', label = 'Samples: real')
  plt.plot(t_s, psi_s.imag, '-g', label = 'Samples: imag')
  plt.xlabel('CL time', fontsize = 20, fontweight = 'bold')
  plt.ylabel('$\psi$', fontsize = 20, fontweight = 'bold') 
  plt.ylim([-5, 5])
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
  
  
  plt.figure(4)
  plt.title('Particle Numbers (real part): CL Simulation', fontsize = 20, fontweight = 'bold')
  plt.plot(t_s, N_up_s.real, '-', color = 'green', label = 'Samples: Up')
  plt.plot(t_s, N_dwn_s.real, '-', color = 'red', label = 'Samples: Down')
  # plt.plot(t_s, np.ones(len(t_s)), 'k', label = 'Constraint')
  plt.xlabel('CL time', fontsize = 20, fontweight = 'bold')
  plt.ylabel('$N$', fontsize = 20, fontweight = 'bold') 
  plt.ylim([-5, 20])
  plt.legend()
  plt.show()
  
  plt.figure(5)
  plt.title('Particle Numbers (imag part): CL Simulation', fontsize = 20, fontweight = 'bold')
  plt.plot(t_s, N_up_s.imag, '-', color = 'green', label = 'Samples: Up')
  plt.plot(t_s, N_dwn_s.imag, '-', color = 'red', label = 'Samples: Down')
  # plt.plot(t_s, np.ones(len(t_s)), 'k', label = 'Constraint')
  plt.xlabel('CL time', fontsize = 20, fontweight = 'bold')
  plt.ylabel('Im($N$)', fontsize = 20, fontweight = 'bold') 
  plt.legend()
  plt.ylim([-5, 20])
  plt.show()

  plt.figure(6)
  plt.title('Magnetization: CL Simulation', fontsize = 20, fontweight = 'bold')
  plt.plot(t_s, Mag_s.real, '-', color = 'purple', label = 'Samples')
  plt.plot(t_s, Mag_s.imag, '-', color = 'skyblue', label = 'Samples')
  # plt.plot(t_s, np.ones(len(t_s)), 'k', label = 'Constraint')
  plt.xlabel('CL time', fontsize = 20, fontweight = 'bold')
  plt.ylabel('$M_{z}$', fontsize = 20, fontweight = 'bold') 
  plt.ylim([-5, 20])
  plt.legend()
  plt.show()
  
  plt.figure(7)
  plt.title('$M^2$ : CL Simulation', fontsize = 20, fontweight = 'bold')
  plt.plot(t_s, M2_s.real, '-', color = 'purple', label = 'Samples')
  plt.plot(t_s, M2_s.imag, '-', color = 'skyblue', label = 'Samples')
  # plt.plot(t_s, np.ones(len(t_s)), 'k', label = 'Constraint')
  plt.xlabel('CL time', fontsize = 20, fontweight = 'bold')
  plt.ylabel('$M^2$', fontsize = 20, fontweight = 'bold') 
  plt.ylim([-5, 20])
  plt.legend()
  plt.show()
  
