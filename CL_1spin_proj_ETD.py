import numpy as np
from numpy import linalg
import yaml
import math
import matplotlib
matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt 
import time
from scipy.fft import fft 
from scipy.fft import ifft
import matplotlib 
import matplotlib.pyplot as plt 
import sys
import copy


# Helper functions

def psi_saddle(_h, n):
	real_part = 2*np.pi*n # n is the saddle point mode index, since the real part is periodic; n is an integer 
	imag_part = -2*np.arctan(0.25*((sech(_h/2))**2)*(np.lib.scimath.sqrt(1 - 2*np.cosh(2*_h)) + 1j))
	psi_star = real_part + imag_part
	return psi_star

def A_nk(n, ntau, beta, alpha, _hz):
	A = 1.
	A += beta * _hz * ((-1)**(alpha)) / ntau
	A *= -np.exp(-2. * np.pi * 1j * n / ntau)
	A += 1.
	return A 

def sech(x):
	return 1/(np.cosh(x))



def fill_forces(phi_up, phi_dwn, phistar_up, phistar_dwn, dSdphistar_up, dSdphistar_dwn, dSdphi_up, dSdphi_dwn, ntau, _psi, _gamma, U):
  dSdphistar_up.fill(0.)
  dSdphistar_dwn.fill(0.)
  dSdphi_up.fill(0.)
  dSdphi_dwn.fill(0.)

  # Build force vector 
  for itau in range(0, ntau):
    # PBC 
    itaum1 = ( (int(itau) - 1) % int(ntau) + int(ntau)) % int(ntau)
    # Build force vector 
    dSdphistar_up[itau] += phi_up[itau] - phi_up[itaum1] + 1j * phi_up[itaum1] * _psi / ntau + phi_up[itaum1] * _gamma/ntau 
    dSdphistar_up[itau] += U * (1./ntau) * phi_up[itaum1] * phi_up[itaum1] * phistar_up[itau]
    dSdphi_up[itaum1] += phistar_up[itaum1] - phistar_up[itau] + 1j * phistar_up[itau] * _psi / ntau + phistar_up[itau] * _gamma/ntau 
    dSdphi_up[itaum1] += U * (1./ntau) * phistar_up[itau] * phi_up[itaum1] * phistar_up[itau]

    dSdphistar_dwn[itau] += phi_dwn[itau] - phi_dwn[itaum1] + 1j * phi_dwn[itaum1] * _psi / ntau + phi_dwn[itaum1] * _gamma/ntau 
    dSdphistar_dwn[itau] += U * (1./ntau) * phi_dwn[itaum1] * phi_dwn[itaum1] * phistar_dwn[itau]
    dSdphi_dwn[itaum1] += phistar_dwn[itaum1] - phistar_dwn[itau] + 1j * phistar_dwn[itau] * _psi/ ntau + phistar_dwn[itau] * _gamma/ntau 
    dSdphi_dwn[itaum1] += U * (1./ntau) * phistar_dwn[itau] * phi_dwn[itaum1] * phistar_dwn[itau]




def step_ETD(Phi, Lincoefs, nonlincoefs, noisescls, nonlin_forces, w_vector, noises, L_vector, ntau):
  # FFT phi's
  # linear term
  phi_up_cp, phistar_up_cp, phi_dwn_cp, phistar_dwn_cp = np.split(Phi, 4)
  noise_up, noisestar_up, noise_dwn, noisestar_dwn = np.split(noises, 4)
  lincoef_up, lincoef_dwn = np.split(Lincoefs, 2)
  nonlincoef_up, nonlincoef_dwn = np.split(nonlincoefs, 2)
  noisescl_up, noisescl_dwn = np.split(noisescls, 2)
  L_up, L_dwn, Lstar_up, Lstar_dwn = np.split(L_vector, 4)
  w_up, w_dwn, w_up_star, w_dwn_star= np.split(w_vector, 4)


  phi_up_cp = fft(phi_up_cp) * lincoef_up
  phi_dwn_cp = fft(phi_dwn_cp) * lincoef_dwn
  phistar_up_cp = fft(phistar_up_cp) * np.conj(lincoef_up 
  phistar_dwn_cp = fft(phistar_dwn_cp) * np.conj(lincoef_dwn)

  # add nonlinear term 
  phi_up_cp += (fft(L_up * w_up) * nonlincoef_up)
  phi_dwn_cp += (fft(L_dwn * w_dwn) * nonlincoef_dwn)
  phistar_up_cp += (fft(Lstar_up * w_up_star) * np.conj(nonlincoef_up))
  phistar_dwn_cp += (fft(Lstar_dwn * w_dwn_star) * np.conj(nonlincoef_dwn))

  
  
  # FFT and Scale by fourier coeff 
  noise_up = fft(noise_up) * noisescl_up 
  noise_dwn = fft(noise_dwn) * noisescl_dwn
  noisestar_up = fft(noisestar_up) * np.conj(noisescl_up) 
  noisestar_dwn = fft(noisestar_dwn) * np.conj(noisescl_dwn) 

  phi_up_cp += noise_up
  phi_dwn_cp += noise_dwn
  phistar_up_cp += noisestar_up
  phistar_dwn_cp += noisestar_dwn
  





def fill_grad_e(phi_up, phi_dwn, phistar_up, phistar_dwn, grad_e): 
  L_up.fill(0.)
  L_dwn.fill(0.)
  Lstar_up.fill(0.)
  Lstar_dwn.fill(0.)


  # Perform index shifts to get the gradient constraint vectors 
  for itau in range(0, ntau):
    # PBC 
    itaum1 = ( (int(itau) - 1) % int(ntau) + int(ntau)) % int(ntau)
    L_up[itau] += phi_up[itaum1]
    L_dwn[itau] += phi_dwn[itaum1]
    Lstar_up[itaum1] += phistar_up[itau]
    Lstar_dwn[itaum1] += phistar_dwn[itau]
  
  grad_e.fill(0.)
  grad_e += np.hstack([Lstar_up, L_up, Lstar_dwn, L_dwn])
  # grad_e += np.hstack([L_up, Lstar_up, L_dwn, Lstar_dwn]) # off-diagonal relaxation
  grad_e *= 1./float(ntau)




def constraint(y, ntau):
  _result = 0. + 1j * 0. # single site, scalar 
  # unpack 
  phi_up_cp, phistar_up_cp, phi_dwn_cp, phistar_dwn_cp = np.split(y, 4)

  for itau in range(0, ntau): 
    # PBC 
    itaum1 = ( (int(itau) - 1) % int(ntau) + int(ntau)) % int(ntau)
    # accumulate density  
    _result += phi_up_cp[itaum1] * phistar_up_cp[itau]
    _result += phi_dwn_cp[itaum1] * phistar_dwn_cp[itau] 
 
  # normalize by ntau 
  _result *= 1./float(ntau)
  _result -= 1. # Spin-1/2 constraint 
  return _result



# Parameters
gamma_shift = 3.25

seed = 0 
np.random.seed(seed)
ntau = 12
# IC = np.ones(ntau, dtype=np.complex_) * (1/(np.sqrt(2)) + 1j*2/(np.sqrt(3))) 
IC = np.ones(ntau, dtype=np.complex_) * (1./(np.sqrt(2.)) + 1j*0.) 
numspecies = 2 
beta = 1.00
# lambda_psi = 0.025
_hz = 0.0
_U = 0.25

dt = 0.0005

numtsteps = 500000
iofreq = 1000 # printing frequency 

num_points = math.floor(numtsteps/iofreq)


# initialize psi at saddle pt
isPsizero = True 
# isPsizero = False 
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
print('Constrained Complex Langevin Sampling')
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
 
 #phi_up = np.random.normal(0, 1.0, ntau)  + 1j*np.random.normal(0, 1.0, ntau) 
 #phi_dwn = np.random.normal(0, 1.0, ntau) + 1j*np.random.normal(0, 1.0, ntau)  
 #phistar_up = np.random.normal(0, 1.0, ntau) + 1j*np.random.normal(0, 1.0, ntau)  
 #phistar_dwn = np.random.normal(0, 1.0, ntau) + 1j*np.random.normal(0, 1.0, ntau) 

 #phi_up = np.ones(ntau, dtype=np.complex_) 
 #phistar_up = 2.*np.ones(ntau, dtype=np.complex_) 
 #phi_dwn = 3.* np.ones(ntau, dtype=np.complex_) 
 #phistar_dwn = 4.*np.ones(ntau, dtype=np.complex_) 
L_up = np.zeros(ntau, dtype=np.complex_)
L_dwn= np.zeros(ntau, dtype=np.complex_)
Lstar_up = np.zeros(ntau, dtype=np.complex_)
Lstar_dwn = np.zeros(ntau, dtype=np.complex_)


dTau = beta/ntau


noise_up = np.zeros(ntau, dtype=np.complex_)
noise_dwn= np.zeros(ntau, dtype=np.complex_)
noisestar_up = np.zeros(ntau, dtype=np.complex_)
noisestar_dwn = np.zeros(ntau, dtype=np.complex_)

# Compute the linear and non-linear coefficients once since they are complex scalars and not a function of the configuration for a single spin in this model 
lincoef_up = np.zeros(ntau, dtype=np.complex_)
lincoef_dwn = np.zeros(ntau, dtype=np.complex_)
nonlincoef_up = np.zeros(ntau, dtype=np.complex_)
nonlincoef_dwn = np.zeros(ntau, dtype=np.complex_)
noisescl_up = np.zeros(ntau, dtype=np.complex_)
noisescl_dwn = np.zeros(ntau, dtype=np.complex_)
dSdphistar_up = np.zeros(ntau, dtype=np.complex_) 
dSdphi_up = np.zeros(ntau, dtype=np.complex_) 
dSdphistar_dwn = np.zeros(ntau, dtype=np.complex_) 
dSdphi_dwn = np.zeros(ntau, dtype=np.complex_) 


# Sampling vectors   

t_s = np.zeros(num_points + 1)
num_iters_s = np.zeros(num_points + 1)
N_tot_s = np.zeros(num_points + 1, dtype=np.complex_)
N_up_s = np.zeros(num_points + 1, dtype=np.complex_)
N_dwn_s = np.zeros(num_points + 1, dtype=np.complex_)
Mag_s = np.zeros(num_points + 1, dtype=np.complex_)
M2_s = np.zeros(num_points + 1, dtype=np.complex_)
psi_s = np.zeros(num_points + 1, dtype=np.complex_)


num_iters_s[0] = 0

# initialize the fictitious time 
t = 0.

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


# Initialize Matrices  
I = np.identity(4*ntau, dtype=np.complex_)
Forces = np.zeros(4*ntau, dtype=np.complex_)
all_noises= np.zeros(4*ntau, dtype=np.complex_)
mobility = np.ones(4*ntau)
mobility *= float(ntau) # scale mobility by Ntau 
state_vector = np.zeros(4*ntau, dtype=np.complex_)

for i in range(0, ntau):
  lincoef_up[i] = A_nk(int(i), ntau, beta, 0, _hz)
  lincoef_dwn[i] = A_nk(int(i), ntau, beta, 1, _hz)

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
num_iters_avg = 0

ctr = 1
# Initialize Matrices
# ntau = 1 
G_matrix = np.zeros(((4*ntau) + 1, (4*ntau)+ 1), dtype=np.complex_)  # 4ntau DOF + psi DOF for single spin 
d_vector = np.zeros(4*ntau + 1, dtype=np.complex_) 

y_iter = np.zeros(4*ntau, dtype=np.complex_) 


# Fill G_matrix identity part
for i in range(0, 4*ntau):
  G_matrix[i][i] = 1.


# grad e
grad_e = np.ones(len(y_iter), dtype=np.complex_)
y_tilde = np.zeros(len(y_iter), dtype=np.complex_) 
y_vector = np.zeros(len(y_iter), dtype=np.complex_) 
y1 = np.zeros(len(y_iter), dtype=np.complex_)

ETOL = 1E-5
max_iterations = 250


# Initialize y_vector with the CS fields 
y_vector += np.hstack([phi_up, phistar_up, phi_dwn, phistar_dwn])

start = time.time()

# Timestep using ETD 
for l in range(0, numtsteps + 1):
  fill_grad_e(phi_up, phi_dwn, phistar_up, phistar_dwn, grad_e)
  fill_forces(phi_up, phi_dwn, phistar_up, phistar_dwn, dSdphistar_up, dSdphistar_dwn, dSdphi_up, dSdphi_dwn, ntau, _psi, gamma_shift, _U)
  Forces.fill(0.)
  Forces += np.hstack([dSdphistar_up, dSdphi_up, dSdphistar_dwn, dSdphi_dwn]) 

  if(len(Forces) != len(all_noises)):
    print('Force and noise vector filled incorrectly -- abort ')
    break


  # Generate Noise terms and scale  
  noise_up = np.random.normal(0, 1., ntau) + 1j * np.random.normal(0, 1., ntau)
  noise_dwn = np.random.normal(0, 1., ntau) + 1j * np.random.normal(0, 1., ntau)
  noisestar_up = np.conj(noise_up) 
  noisestar_dwn = np.conj(noise_dwn)
  all_noises.fill(0.) 
  all_noises += np.hstack([noise_up, noisestar_up, noise_dwn, noisestar_dwn]) 
  all_noises *= np.sqrt(mobility * dt)

  # Set up Newton's iteration
  # Use previous psi as initial iterate guess for psi  
  psi_iter = 0. + 1j * 0.
  psi_iter += _psi
  # initial iterate/guess: phi_vector EM stepped with psi term (typical CL)
  y_iter.fill(0.) # reset to zero 
  y_iter += y_vector
  y_iter -= mobility * dt * Forces

  for j in range(0, max_iterations + 1):
    # Do the Naive Euler step
    # calculate y_tilde 
    y_tilde.fill(0.)
    y_tilde += y_vector 

    # unpack y_vector  
    # Unpack to do operator calculations and block averaging  
    phi_up, phistar_up, phi_dwn, phistar_dwn = np.split(y_vector, 4)
    fill_grad_e(phi_up, phi_dwn, phistar_up, phistar_dwn, grad_e);
    y_tilde += grad_e * psi_iter  # current psi iteration 
   
    # Do Euler maruyama on y_tilde
    phi_up, phistar_up, phi_dwn, phistar_dwn = np.split(y_tilde, 4)
    fill_forces(phi_up, phi_dwn, phistar_up, phistar_dwn, dSdphistar_up, dSdphistar_dwn, dSdphi_up, dSdphi_dwn, ntau, 0., gamma_shift, _U)
    Forces = np.hstack([dSdphistar_up, dSdphi_up, dSdphistar_dwn, dSdphi_dwn]) 
    y_EM = y_tilde - mobility * dt * Forces  + all_noises 
    
    # reset size of y1 vector 
    # fill it with the next iteration  
    y1.fill(0.) # zero 
    y1 += y_iter
 
    d_vector.fill(0.) 
    d_vector[-1] += constraint(y1, ntau)  # last entry
    # Evalute grad e at y1  
    phi_up, phistar_up, phi_dwn, phistar_dwn = np.split(y1, 4)
    fill_grad_e(phi_up, phi_dwn, phistar_up, phistar_dwn, grad_e);

    y1 -= (grad_e * psi_iter)
    # Take EM step 
    # phi_vector -= mobility * dt * (dS_dphi(phi_vector, _beta1, _beta2))
    y1 -= y_EM # residuals 

    # Where is grad_e evaluated? either y_iter or y_1 or y_EM ?  
    # Fill G_matrix (only needs to be done once because it's a linear problem)
    phi_up, phistar_up, phi_dwn, phistar_dwn = np.split(y_iter, 4)
    fill_grad_e(phi_up, phi_dwn, phistar_up, phistar_dwn, grad_e);
    for i in range(0, 4*ntau): 
      G_matrix[i][-1] = -2. # final column 
      G_matrix[i][-1] *= grad_e[i] 
      G_matrix[-1][i] = 1.   # final row 
      G_matrix[-1][i] *= grad_e[i] 

    G_matrix[-1][-1] = 0. 

    d_vector[0:4*ntau] += y1 
    
    d_vector *= -1. # "b" vector to do Ax = b
    # do Ax = b, using G as A and delta as b 
    solution = np.linalg.solve(G_matrix, d_vector)
    
    # Update iteration with the solution (updates) 
    y_iter += solution[0:4*ntau]  
    psi_iter += solution[-1]

    if(np.all(solution < ETOL)):
      num_iters_avg += j/iofreq
      num_iters = j
      break;  

  if( j == max_iterations):
    print('Warning: Max iterations exceeded')
    num_iters_avg += max_iterations/iofreq
    num_iters = max_iterations


  # Update the fields, post Newton's iteration 
  y_vector.fill(0.)
  _psi = 0. + 1j * 0.
  _psi += psi_iter
  y_vector += y_iter
 
  # Unpack to do operator calculations and block averaging  
  phi_up, phistar_up, phi_dwn, phistar_dwn = np.split(y_vector, 4)

  # L and L-star vectors represent the appropriately-tau-shifted CSfields
  # Calculate the scalar inverse of G:
  # G_inv_mod = 0.
 #  G_inv_mod = sum((L_up**2) + (L_dwn**2) + (Lstar_up**2) + (Lstar_dwn**2))
 #  G_inv_mod = 1./G_inv_mod

  # Build G matrix
  # G_matrix = np.outer(grad_e, grad_e)
 #  print('Printing unscaled G-matrix')
 #  print(G_matrix)
  # G_matrix *= G_inv_mod # scaled by a norm of sorts 

  # Build projection matrix 
  # Proj_t = I - G_matrix 

  
 #  noise_up = np.array(noise_up)
 #  noise_dwn = np.array(noise_dwn)
 #  noisestar_up = np.array(noisestar_up)
 #  noisestar_dwn = np.array(noisestar_dwn)

 #  proj_noise = Proj_t @ all_noises
 #  proj_noise *= np.sqrt(dt * ntau) 
 #  
 #  proj_force = Proj_t @ Forces 
 #  proj_force *= -dt 
 #  
 #  state_vector = proj_force   
 #  state_vector += proj_noise

  # Step the phi/phistar fields 
 #  phi_up += state_vector[0:ntau]
 #  phistar_up += state_vector[ntau:2*ntau]
 #  phi_dwn += state_vector[2*ntau:3*ntau]
 #  phistar_dwn += state_vector[3*ntau:4*ntau]
 
  # print('Gradient vector: \n')
  # print(grad_e) 
 #  for itau in range(0, ntau):
 #    # Periodic Boundary conditions -- index shifts  
 #    itaup1 = ( int(itau) + 1 ) % int(ntau) 
 #    itaum1 = ( (int(itau)-1) % int(ntau) + int(ntau)) % int(ntau)
 #
 #    # filling algorithm
 #    # 1. diagonal entries first 
 #    # 2. row 
 #    # 3. column
 #
 #    # Diagonals (may be redundent)
 #    G_matrix[itau][itau] = Lstar_up[itaup1] 
 #    G_matrix[itau + ntau][itau + ntau] = L_up[itaum1] 
 #    G_matrix[itau + 2*ntau][itau + 2*ntau] = Lstar_dwn[itaup1] 
 #    G_matrix[itau + 3*ntau][itau + 3*ntau] = L_dwn[itaum1] 
 #
 #    # extend the entry in each diagonal to the right and down 
 #    G_matrix[itau][itau:4*ntau] = Lstar_up[itaup1] 
 #    G_matrix[itau:4*ntau][itau] = Lstar_up[itaup1]
 #
 #    G_matrix[ntau + itau][ntau + itau:4*ntau] = L_up[itaum1] 
 #    G_matrix[ntau + itau:4*ntau][ntau + itau] = L_up[itaum1] 
 #
 #    G_matrix[2 * ntau + itau][2*ntau + itau:4*ntau] = Lstar_dwn[itaup1] 
 #    G_matrix[2*ntau + itau:4*ntau][2*ntau + itau] = Lstar_dwn[itaup1]
 #
 #    G_matrix[3 * ntau + itau][3*ntau + itau:4*ntau] = L_dwn[itaum1] 
 #    G_matrix[3*ntau + itau:4*ntau][3*ntau + itau] = L_dwn[itaum1] 
 #  
 #  print()
 #  print('Printing G-matrix')
 #  print(G_matrix)
 # #
 #  print() 
 #  print('Projection matrix: \n') 
 #  print(Proj_t) 
 #
 #  print() 
 #  print('Proj * Proj: \n') 
 #  print(Proj_t @ Proj_t) 
 #  print() 
 #  print('Proj*Proj  - Proj: \n') 
 #  print((Proj_t @ Proj_t )- Proj_t) 
  # break

  # linear term
 #  phi_up = fft(phi_up) * lincoef_up 
 #  phi_dwn = fft(phi_dwn) * lincoef_dwn 
 #  phistar_up = fft(phistar_up) * np.conj(lincoef_up) 
 #  phistar_dwn = fft(phistar_dwn) * np.conj(lincoef_dwn)
 #
 #  # add nonlinear term 
 #  phi_up += (fft(L_up * nonlinforce) * nonlincoef_up)
 #  phi_dwn += (fft(L_dwn * nonlinforce) * nonlincoef_dwn)
 #  phistar_up += (fft(Lstar_up * nonlinforce) * np.conj(nonlincoef_up))
 #  phistar_dwn += (fft(Lstar_dwn * nonlinforce) * np.conj(nonlincoef_dwn))

  
  
  # FFT and Scale by fourier coeff 
 #  noise_up = fft(noise_up) * noisescl_up 
 #  noise_dwn = fft(noise_dwn) * noisescl_dwn
 #  noisestar_up = fft(noisestar_up) * np.conj(noisescl_up) 
 #  noisestar_dwn = fft(noisestar_dwn) * np.conj(noisescl_dwn) 
  # Euler step
 #  phi_up -= ntau * dt * dSdphistar_up 
 #  phistar_up -= ntau * dt * dSdphi_up 
 #  phi_dwn -= ntau * dt * dSdphistar_dwn 
 #  phistar_dwn -= ntau * dt * dSdphi_dwn 
 #  
 #  phi_up += (noise_up * np.sqrt(dt * ntau)) 
 #  phi_dwn += (noise_dwn * np.sqrt(dt * ntau)) 
 #  phistar_up += (noisestar_up * np.sqrt(dt * ntau)) 
 #  phistar_dwn += (noisestar_dwn * np.sqrt(dt * ntau)) 

  # inverse fft  
 #  phi_up = ifft(phi_up) * ntau
 #  phi_dwn = ifft(phi_dwn) * ntau
 #  phistar_up = ifft(phistar_up) * ntau
 #  phistar_dwn = ifft(phistar_dwn) * ntau
 #
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
    print('Trajectory diverged at dt iteration: ' + str(l) + ' and CL time = ' + str(t))
    break

 
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
    num_iters_s[ctr] = num_iters_avg 
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
    num_iters_avg = 0.
    ctr += 1


if(l != numtsteps): 
  divergence_index = np.where(N_tot_s.real == 0)
  divergence_index = divergence_index[0][0]
  t_s = t_s[0:divergence_index]
  N_tot_s = N_tot_s[0:divergence_index]
  N_up_s = N_up_s[0:divergence_index]
  N_dwn_s = N_dwn_s[0:divergence_index]
  Mag_s = Mag_s[0:divergence_index]
  M2_s = M2_s[0:divergence_index]
  num_iters_s = num_iters_s[0:divergence_index]
  psi_s = psi_s[0:divergence_index]



end = time.time()
print()
print()
print('Simulation finished: Runtime = ' + str(end - start) + ' seconds')

# Print the results (noise long-time averages)
print()
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

plt.figure(13)
plt.title('Iterations per timestep sampling: CL Simulation', fontsize = 20, fontweight = 'bold')
plt.plot(t_s, num_iters_s, '-r', label = 'Samples')
plt.xlabel('CL time', fontsize = 20, fontweight = 'bold')
plt.ylabel('Iterations', fontsize = 20, fontweight = 'bold') 
# plt.ylim([-5, 5])
plt.legend()
plt.show()
