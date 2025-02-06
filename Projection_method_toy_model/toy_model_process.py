import csv
from mpmath import *
import subprocess
import os
import re
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pdb
import yaml
import math
## This function runs statistics on the runs accessed (i.e. parameter sweep). Then it collects the relevant data and plots it at the end

def sech(x):
  return 1/(np.cosh(x))


gamma=[-2.5, -2.25, -2.0, -1.75, -1.5, -1.25, -1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5] 

with open('params.yml') as infile:
  master_params = yaml.load(infile, Loader=yaml.FullLoader)

# Get the reference parameters for these sweeps, i.e. the constant parameters kept throughout the runs (tau, v, etc.)
mu = master_params['model']['mu']
h = master_params['model']['h']
dt = master_params['timestepping']['dt']
ntmax = master_params['timestepping']['ntmax']


runlength_max = ntmax * dt
apply_ADT = False
output_file_names = ['No_psi', 'projected', 'psi'] 

# Observables  
runtime = np.zeros((len(gamma), len(output_file_names)))
ReTrU = np.zeros((len(gamma), len(output_file_names)))
errReTrU = np.zeros((len(gamma), len(output_file_names)))
ImTrU = np.zeros((len(gamma), len(output_file_names)))
errImTrU = np.zeros((len(gamma), len(output_file_names)))

ReTrU2 = np.zeros((len(gamma), len(output_file_names)))
errReTrU2 = np.zeros((len(gamma), len(output_file_names)))
ImTrU2 = np.zeros((len(gamma), len(output_file_names)))
errImTrU2 = np.zeros((len(gamma), len(output_file_names)))

ReTrUUdag = np.zeros((len(gamma), len(output_file_names)))
errReTrUUdag = np.zeros((len(gamma), len(output_file_names)))
ImTrUUdag = np.zeros((len(gamma), len(output_file_names)))
errImTrUUdag = np.zeros((len(gamma), len(output_file_names)))

Re_constraint = np.zeros((len(gamma), len(output_file_names)))
errRe_constraint = np.zeros((len(gamma), len(output_file_names)))
Im_constraint = np.zeros((len(gamma), len(output_file_names)))
errIm_constraint = np.zeros((len(gamma), len(output_file_names)))



for i, gamma_ in enumerate(gamma):
  pwd = "."
  path = pwd + "/" + "gamma_" + str(gamma_) 
  print(path)

  # 3 files to process
  for k, name in enumerate(output_file_names):
    filename = 'data_' + name + '.dat'  
    print('processing ' + filename)
    data_file = 'averages_' + name + '.dat'
    # run processing script       
    cmd_string = "python3 ~/csbosonscpp/tools/stats.py -f " + path + "/" + filename + " -o Tr_U.real Tr_U.imag Tr_U2.real Tr_U2.imag Tr_UUdag.real Tr_UUdag.imag constraint.real constraint.imag -a -q > " + path + "/" + data_file 
    check_path = path + '/' + data_file 
        
    # Check to see whether we have any runlength
    cols = np.loadtxt(path + "/" + filename, unpack=True)
    t = cols[0] # first column is time
    if(len(t) > 1): 
      runtime[i, k] = t[-2]
    else:
      runtime[i, k] = t[0] 

    if not os.path.exists(check_path):
      if (runtime[i, k] == 0):
        print("No runtime, inserting NaN for observables")
        ReTrU[i,k] = math.nan 
        errReTrU[i,k] = math.nan 
        ImTrU[i,k] = math.nan 
        errImTrU[i,k] = math.nan 
        
        ReTrU2[i,k] = math.nan 
        errReTrU2[i,k] = math.nan 
        ImTrU2[i,k] = math.nan 
        errImTrU2[i,k] = math.nan 
        
        ReTrUUdag[i,k] = math.nan 
        errReTrUUdag[i,k] = math.nan 
        ImTrUUdag[i,k] = math.nan 
        errImTrUUdag[i,k] = math.nan 
        
        Re_constraint[i,k] = math.nan 
        errRe_constraint[i,k] = math.nan 
        Im_constraint[i,k] = math.nan 
        errIm_constraint[i,k] = math.nan 
      else:
        subprocess.call(cmd_string, shell = True)

      if (runtime[i, k] != 0):
        in_file = open(path + "/" + data_file, "r")
        tmp = in_file.read()
        tmp = re.split(r"\s+", tmp)
        tmp = tmp[0:-1]
        tmp = tuple(map(float, tmp))

        ReTrU[i,k] = tmp[0] 
        errReTrU[i,k] = tmp[1] 
        ImTrU[i,k] = tmp[2] 
        errImTrU[i,k] = tmp[3] 
        
        ReTrU2[i,k] = tmp[4] 
        errReTrU2[i,k] = tmp[5] 
        ImTrU2[i,k] = tmp[6] 
        errImTrU2[i,k] = tmp[7] 
        
        ReTrUUdag[i,k] = tmp[8] 
        errReTrUUdag[i,k] = tmp[9] 
        ImTrUUdag[i,k] = tmp[10] 
        errImTrUUdag[i,k] = tmp[11] 
        
        Re_constraint[i,k] = tmp[12] 
        errRe_constraint[i,k] = tmp[13] 
        Im_constraint[i,k] = tmp[14] 
        errIm_constraint[i,k] = tmp[15] 
        in_file.close()




# Plot Tr(U)
plt.figure(1)
for k, name in enumerate(output_file_names):
  plt.errorbar(gamma, ReTrU[:, k], errReTrU[:, k], linewidth=0.5, markersize=6, marker = '*', label = name)
# plt.plot(U, np.ones(len(U))*np.tanh(0.2), linewidth=1, label = 'exact')
plt.title('CL Methods Comparison', fontsize = 11)
plt.xlabel('$\gamma$', fontsize = 20, fontweight = 'bold')
plt.ylabel('Tr[$U$]', fontsize = 20, fontweight = 'bold')
plt.legend()
plt.show()

plt.figure(2)
for k, name in enumerate(output_file_names):
  plt.errorbar(gamma, ReTrU2[:, k], errReTrU2[:, k], linewidth=0.5, markersize=6, marker = '*', label = name)
plt.title('CL Methods Comparison', fontsize = 11)
plt.xlabel('$\gamma$', fontsize = 20, fontweight = 'bold')
plt.ylabel('Tr[$U^2$]', fontsize = 20, fontweight = 'bold')
plt.legend()
plt.show()



plt.figure(3)
for k, name in enumerate(output_file_names):
  plt.plot(gamma, runtime[:, k], linewidth=0.5, markersize=6, marker = '*', label = name)
#plt.plot(U, np.ones(len(U))*1.00, linewidth=1, label = 'exact')
plt.title('CL Methods Comparison: Stability', fontsize = 11)
plt.xlabel('$\gamma$', fontsize = 20, fontweight = 'bold')
plt.ylabel('Runtime', fontsize = 20, fontweight = 'bold')
#plt.ylim(-1,1)
#plt.xscale('log')
plt.legend()
#plt.savefig('Mag_Test.eps', dpi = 300)
#plt.savefig("plt/"+args.title[0]+'_'+str(i)+'.png', dpi=300) #dpi=72
plt.show()



