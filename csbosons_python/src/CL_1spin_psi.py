import numpy as np
import yaml
import math
import matplotlib
matplotlib.rcParams['text.usetex'] = True
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt 
import time
from scipy.fft import fft 
from scipy.fft import ifft
import matplotlib 
import matplotlib.pyplot as plt 



# Helper functions

def psi_saddle(_h, n):
	real_part = 2*np.pi*n # n is the saddle point mode index, since the real part is periodic; n is an integer 
	imag_part = -2*np.arctan(0.25*((sech(_h/2))**2)*(np.lib.scimath.sqrt(1 - 2*np.cosh(2*_h)) + 1j))
	psi_star = real_part + imag_part
	return psi_star

def A_nk(n, ntau, beta, alpha, h, U):
	A = 1.
	A += beta * h * ((-1)**(alpha)) / ntau
	A += U * beta / (2. * ntau) 
	A *= -np.exp(-2. * np.pi * 1j * n / ntau) 
	A += 1.
	return A 

def sech(x):
	return 1/(np.cosh(x))


# Parameters 
pcnt_noise = 1.
ntau = 16
# IC = np.ones(ntau, dtype=np.complex_) * (1/(np.sqrt(2)) + 1j*2/(np.sqrt(3))) 
IC = np.ones(ntau, dtype=np.complex_) * (1/(np.sqrt(2)) + 1j*0) 
numspecies = 2 
beta = 1.0
lambda_psi = 0.05
# _gamma = 2.0
_hz = 0.0
_U = 0.00

dt = 0.0015

numtsteps = 500000
iofreq = 1500 # print every 1000 steps 

num_points = math.floor(numtsteps/iofreq)


# initialize psi at saddle pt
isPsizero = False 
_psi = 0. + 1j * 0. 
if(isPsizero):
  _psi = 0.
else:
  _psi += psi_saddle(_hz, int(0)) # choose 0th mode 

print()
print()
print('-----Single Spin simulation, Schwinger Bosonic Coherent States----')
print()
print()
print()
print('Applied Magnetic Field : ' + str(_hz))
print()
print('Soft Repulsion Potential Strength: ' + str(_U))
print()
print('Temperature : ' + str(1/beta) + ' Kelvin')
print()
print('Running for ' + str(numtsteps) + ' timesteps')
print()
print('Using Ntau = ' + str(ntau) + ' tau slices' )
print()
if(isPsizero):
  print('Pinning psi to zero')
else:
  print('Intializing psi at its saddle point: ')

print(_psi)
print()
print('Complex Langevin Sampling')
print()

# initialize CS fields at zero 
phi_up = np.zeros(ntau, dtype=np.complex_)
phi_dwn = np.zeros(ntau, dtype=np.complex_)
phistar_up = np.zeros(ntau, dtype=np.complex_)
phistar_dwn = np.zeros(ntau, dtype=np.complex_)

# Fill with initial condition 
phi_up += IC
phi_dwn += IC
phistar_up += IC
phistar_dwn += IC

# option to do normally distributed random nums
 #phi_up = np.random.normal(0, 1.0, ntau) 
 #phi_dwn = np.random.normal(0, 1.0, ntau) 
 #phistar_up = np.random.normal(0, 1.0, ntau) 
 #phistar_dwn = np.random.normal(0, 1.0, ntau)


dTau = beta/ntau


noise_up = np.zeros(ntau, dtype=np.complex_)
noise_dwn= np.zeros(ntau, dtype=np.complex_)
noisestar_up = np.zeros(ntau, dtype=np.complex_)
noisestar_dwn = np.zeros(ntau, dtype=np.complex_)

L_up = np.zeros(ntau, dtype=np.complex_)
L_dwn= np.zeros(ntau, dtype=np.complex_)
Lstar_up = np.zeros(ntau, dtype=np.complex_)
Lstar_dwn = np.zeros(ntau, dtype=np.complex_)

# Compute the linear and non-linear coefficients once since they are complex scalars and not a function of the configuration for a single spin in this model 
lincoef_up = np.zeros(ntau, dtype=np.complex_)
lincoef_dwn = np.zeros(ntau, dtype=np.complex_)
nonlincoef_up = np.zeros(ntau, dtype=np.complex_)
nonlincoef_dwn = np.zeros(ntau, dtype=np.complex_)
noisescl_up = np.zeros(ntau, dtype=np.complex_)
noisescl_dwn = np.zeros(ntau, dtype=np.complex_)


# Sampling vectors   

t_s = np.zeros(num_points + 1)
N_tot_s = np.zeros(num_points + 1, dtype=np.complex_)
N_up_s = np.zeros(num_points + 1, dtype=np.complex_)
N_dwn_s = np.zeros(num_points + 1, dtype=np.complex_)
Mag_s = np.zeros(num_points + 1, dtype=np.complex_)
M2_s = np.zeros(num_points + 1, dtype=np.complex_)
psi_s = np.zeros(num_points + 1, dtype=np.complex_)


# Calculate the particle numbers
N_up = 0.
N_dwn = 0.
N_tot = 0. 
for itau in range(0, int(ntau)):
  itaum1 = ( (int(itau) - 1) % int(ntau) + int(ntau)) % int(ntau)
  N_up += phistar_up[itau] * phi_up[itaum1]
  N_dwn += phistar_dwn[itau] * phi_dwn[itaum1]
  
# scale by ntau
N_up *= 1./ntau 
N_dwn *= 1./ntau 

N_tot = N_up + N_dwn
Mag = N_up - N_dwn
M2 = Mag**2 

psi_s[0] = _psi
N_tot_s[0] = N_tot
N_up_s[0] = N_up
N_dwn_s[0] = N_up
Mag_s[0] = Mag
M2_s[0] = M2

# initialize the fictitious time 
t = 0.

for i in range(0, ntau):
  lincoef_up[i] = A_nk(int(i), ntau, beta, 0, _hz, _U)
  lincoef_dwn[i] = A_nk(int(i), ntau, beta, 1, _hz, _U)

  # Correct diverging terms by using Euler limit of ETD 
  if(lincoef_up[i] == 0.):
    nonlincoef_up[i] = (1./ntau) * -1. * ntau * dt 
    noisescl_up[i] = (1./ntau) * np.sqrt(ntau * dt)
  else: 
    nonlincoef_up[i] = (1./ntau) * (np.exp(-lincoef_up[i] * ntau * dt) - 1.)/lincoef_up[i]
    noisescl_up[i] = (1./ntau) * np.sqrt((1. - np.exp(-2. * lincoef_up[i] * ntau * dt))/(2. * lincoef_up[i]))

  if(lincoef_dwn[i] == 0.):
    nonlincoef_dwn[i] = (1./ntau) * -1. * ntau * dt 
    noisescl_dwn[i] = (1./ntau) * np.sqrt(ntau * dt)
  else: 
    nonlincoef_dwn[i] = (1./ntau) * (np.exp(-lincoef_dwn[i] * ntau * dt) - 1.)/lincoef_dwn[i]
    noisescl_dwn[i] = (1./ntau) * np.sqrt((1. - np.exp(-2. * lincoef_dwn[i] * ntau * dt))/(2. * lincoef_dwn[i]))

  lincoef_up[i] = np.exp(- lincoef_up[i] * ntau * dt)
  lincoef_up[i] *= (1./ntau) 
  lincoef_dwn[i] = np.exp(- lincoef_dwn[i] * ntau * dt)
  lincoef_dwn[i] *= (1./ntau) 

N_tot_avg = 0. + 1j*0 
N_up_avg = 0. + 1j*0 
N_dwn_avg = 0. + 1j*0 
M_avg = 0. + 1j*0
M2_avg = 0. + 1j*0

ctr = 1


start = time.time()

# Timestep using ETD 
for l in range(0, numtsteps + 1):
  L_up.fill(0.)
  L_dwn.fill(0.)
  Lstar_up.fill(0.)
  Lstar_dwn.fill(0.)

  # Perform index shifts to get N, N* vectors  TODO : this is wrong for some reason, but it should be correct 
  for itau in range(0, ntau):
    # PBC 
    itaum1 = ( (int(itau) - 1) % int(ntau) + int(ntau)) % int(ntau)
    L_up[itau] += phi_up[itaum1]
    L_dwn[itau] += phi_dwn[itaum1]
    Lstar_up[itaum1] += phistar_up[itau]
    Lstar_dwn[itaum1] += phistar_dwn[itau]

  nonlinforce = 1j * _psi / ntau 
  # nonlinforce = (1j * _psi / ntau) + (dTau * _U * np.sinh(N_tot - 1.))
  # linear term
  # FFT the vectors   
  phi_up = fft(phi_up) * lincoef_up 
  phi_dwn = fft(phi_dwn) * lincoef_dwn 
  phistar_up = fft(phistar_up) * np.conj(lincoef_up) 
  phistar_dwn = fft(phistar_dwn) * np.conj(lincoef_dwn)

  # add nonlinear term 
  phi_up += (fft(L_up * nonlinforce) * nonlincoef_up)
  phi_dwn += (fft(L_dwn * nonlinforce) * nonlincoef_dwn)
  phistar_up += (fft(Lstar_up * nonlinforce) * np.conj(nonlincoef_up))
  phistar_dwn += (fft(Lstar_dwn * nonlinforce) * np.conj(nonlincoef_dwn))

  # Generate Noise terms 
  noise_up = np.random.normal(0, 1., ntau) + 1j * np.random.normal(0, 1., ntau)
  noise_dwn = np.random.normal(0, 1., ntau) + 1j * np.random.normal(0, 1., ntau)
  noisestar_up = np.conj(noise_up) 
  noisestar_dwn = np.conj(noise_dwn) 
  
  noise_up = np.array(noise_up)
  noise_dwn = np.array(noise_dwn)
  noisestar_up = np.array(noisestar_up)
  noisestar_dwn = np.array(noisestar_dwn)
  
  # FFT and Scale by fourier coeff 
  noise_up = fft(noise_up) * noisescl_up 
  noise_dwn = fft(noise_dwn) * noisescl_dwn
  noisestar_up = fft(noisestar_up) * np.conj(noisescl_up) 
  noisestar_dwn = fft(noisestar_dwn) * np.conj(noisescl_dwn) 
  
  phi_up += noise_up 
  phi_dwn += noise_dwn 
  phistar_up += noisestar_up 
  phistar_dwn += noisestar_dwn 

  # inverse fft  
  phi_up = ifft(phi_up) * ntau
  phi_dwn = ifft(phi_dwn) * ntau
  phistar_up = ifft(phistar_up) * ntau
  phistar_dwn = ifft(phistar_dwn) * ntau

  # Calculate the particle numbers
  N_up = 0.
  N_dwn = 0.
  N_tot = 0. 
  for itau in range(0, int(ntau)):
    itaum1 = ( (int(itau) - 1) % int(ntau) + int(ntau)) % int(ntau)
    N_up += phistar_up[itau] * phi_up[itaum1]
    N_dwn += phistar_dwn[itau] * phi_dwn[itaum1]
  
  # scale by ntau
  N_up *= 1./ntau 
  N_dwn *= 1./ntau 

  N_tot = N_up + N_dwn
  Mag = N_up - N_dwn
  M2 = Mag**2 

 
  if(np.isnan(M2)):
    print('Trajectory diverged at iteration: ' + str(l) + ' and CL time = ' + str(t))
    break

  # Step the psi field
  if(isPsizero):
    _psi = 0.
  else:
    _psi -= 1j * (lambda_psi * dt) * (N_tot - 1.)
    # Add the psi noise 
    psi_noisescl = np.sqrt(2. * lambda_psi * dt) 
    eta = np.random.normal() * psi_noisescl 
    _psi += eta * pcnt_noise

 
  # Calculate observables - sample   
  N_tot_avg += N_tot/iofreq 
  N_up_avg += N_up/iofreq
  N_dwn_avg += N_dwn/iofreq
  M_avg += Mag/iofreq
  M2_avg += M2/iofreq

  t += dt

  # Output on interval
  if(l % iofreq == 0 and l > 0):
     if(ctr %  25):
       print("Completed {} of {} steps".format(l, numtsteps))
     # opout.write("{} {} {} {} {}\n".format(it, Msum.real / Navg, Msum.imag / Navg, psi.real, psi.imag))
     t_s[ctr] = t 
     Mag_s[ctr] = M_avg 
     N_tot_s[ctr] = N_tot_avg 
     N_up_s[ctr] = N_up_avg 
     N_dwn_s[ctr] = N_dwn_avg 
     M2_s[ctr] = M2_avg
     psi_s[ctr] = _psi
     # clear the averages 
     M_avg = 0. + 1j*0 
     M2_avg = 0. + 1j*0 
     N_tot_avg = 0. + 1j*0 
     N_up_avg = 0. + 1j*0 
     N_dwn_avg = 0. + 1j*0 
     ctr += 1

    

end = time.time()
print()
print()
print('Simulation finished: Runtime = ' + str(end - start) + ' seconds')




# Print the results (noise long-time averages)
print()
print()
print('The Particle Number is: ' + str(np.mean(N_tot_s[10:].real)))
print()
print('The Up Boson Particle Number is: ' + str(np.mean(N_up_s[10:].real)))
print()
print('The Down Boson Particle Number is: ' + str(np.mean(N_dwn_s[10:].real)))
print()
print('The Magnetization is: ' + str(np.mean(Mag_s[10:].real)))
print()
print('The Magnetization-squared is: ' + str(np.mean(M2_s[10:].real)))


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

