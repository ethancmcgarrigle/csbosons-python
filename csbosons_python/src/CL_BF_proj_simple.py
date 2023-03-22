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


# Python function for executing CL sampling of a bosefluid in the coherent states field theoretic representation 

# This will be used to test numerical projection methods 

# 3. confirm projected implementation -- In progress  

# Helper functions


# Refresh linear force coefficient
def A_nk(n, _ntau, _beta, k2_grid, _lambda, _ensemble, _mu):
        # Returns a d-dimensional field, evaluated at timeslice "n" 
        #A = np.zeros(len(k2_grid), dtype=np.complex_) # spatial resolution  
        A = 0. 
        dtau = _beta/_ntau
        A += (- dtau * _lambda * k2_grid) # hbar^2/2m k^2, kinetic energy 
        if(_ensemble == "GRAND"):
          A += (_mu * dtau)
        A += 1. 
        A *= -np.exp(-2. * np.pi * 1j * n / _ntau) 
        A += 1.
        return A 


def fill_forces(phi, phistar, dSdphistar, dSdphi, ntau, _psi, _ensemble, _g, _beta):
  # Fills the nonlinear forces as applicable  
  dSdphistar.fill(0.)
  dSdphi.fill(0.)

  dtau = _beta/ntau
  # Build force vector 
  for itau in range(0, int(ntau)):
    # PBC 
    itaum1 = ( (int(itau) - 1) % int(ntau) + int(ntau)) % int(ntau)
    # Build force vector 
    dSdphistar[:, itau] += _g * dtau * phi[:, itaum1] * phi[:, itaum1] * phistar[:, itau]
    dSdphi[:, itaum1] += _g * dtau * phistar[:, itau] * phi[:, itaum1] * phistar[:, itau]
    if(_ensemble == "CANONICAL"):
      # include mu force in nonlinear
      # identify _mu = i psi / \beta
      _mu_eff = 1j * _psi / _beta
      #print('printing the effective chemical potential')
      #print(_mu_eff)
      dSdphistar[:, itau] += -phi[:, itaum1] * _mu_eff * dtau 
      dSdphi[:, itaum1] +=  -phistar[:, itau] * _mu_eff * dtau 





def fill_grad_e(phi, phistar,  grad_e): 
  L =  np.zeros((Nx**dim, ntau), dtype=np.complex_) 
  Lstar =  np.zeros((Nx**dim, ntau), dtype=np.complex_) 
  L.fill(0.)
  Lstar.fill(0.)

  # Perform index shifts to get the gradient constraint vectors 
  for itau in range(0, ntau):
    # PBC 
    itaum1 = ( (int(itau) - 1) % int(ntau) + int(ntau)) % int(ntau)
    L[:, itau] += phi[:, itaum1]
    Lstar[:, itaum1] += phistar[:, itau]
  
  grad_e.fill(0.)
  grad_e += np.hstack([Lstar[0, :], L[0, :]])
  grad_e *= 1./float(ntau)



def constraint_err(_N_input, phi, phistar):
    # Function for calculation the constraint error
    tmp = np.zeros(len(phi[:,0]), dtype=np.complex_)
    Ntau = len(phi[0, :]) 
    for itau in range(0, Ntau):
      itaum1 = ( (int(itau) - 1) % int(Ntau) + int(Ntau)) % int(Ntau)
      tmp += phistar[:, itau] * phi[:, itaum1]
    tmp *= 1./Ntau

    constraint_residual = 0.0 + 1j*0.0
    constraint_residual = integrate_r_intensive(tmp) * Vol
    constraint_residual -= _N_input
    return constraint_residual  # should be near zero 


def integrate_r_intensive(field):
    N_spatial = len(field)
    result = np.sum(field)/N_spatial
    return result


def ETD(_phi, _phistar, _dSdphistar, _dSdphi, _lincoef, _nonlincoef, _noise, _noisestar):
    # Performs Exponential time-differncing time stepping, returns updated fields in real space, \tau representation 
    ntau = len(_phi[0, :])
    _N_spatial = len(_phi)
    # Exponential-Time-Differencing, assumes off-diagonal stepping 

    # Function to step phi and phistar with ETD  
    _phi = fft_dp1(_phi) * _lincoef 
    _phistar = fft_dp1(_phistar) * np.conj(_lincoef) 

    # add nonlinear term, off-diagonal relaxation 
    _phi += (fft_dp1(_dSdphistar) * _nonlincoef)
    _phistar += (fft_dp1(_dSdphi) * np.conj(_nonlincoef))

    # Add the noise
    _phi += _noise 
    _phistar += _noisestar 

    # inverse fft  
    _phi = ifft_dp1(_phi) 
    _phistar = ifft_dp1(_phistar) 

    return [_phi, _phistar]
    # Return state vector (packaged phi/phistar vector)



def EM(phi, phistar, EM_force_phi, EM_force_phistar, _isOffDiagonal):
    # Function to step phi and phistar with ETD  
    noise.fill(0.) 
    noisestar.fill(0.) 

    if(_isOffDiagonal):
      phi -= EM_force_phistar * mobility * dt 
      phistar -= EM_force_phi * mobility * dt 
      noise = np.random.normal(0, 1., (N_spatial, ntau)) + 1j * np.random.normal(0, 1., (N_spatial, ntau))
      noisestar = np.conj(noise) 
    else:
      phi -= EM_force_phi * mobility * dt 
      phistar -= EM_force_phistar * mobility * dt
      # For diagonal stepping, generate real noise  
      noise = np.random.normal(0, 1., (N_spatial, ntau))  
      noisestar = np.random.normal(0, 1., (N_spatial, ntau)) 

    # Add the noise 
    phi += _noise 
    phistar += _noisestar 

    return [phi, phistar]




# d+1 Fourier Transforms --- passed test check for normalization  
def fft_dp1(_CSfield, _dontScale=True):
    # d+1 Fourier transform of a CS field object -- 2D numpy array \in C^ (Nx**dim , ntau)
    ntau = len(_CSfield[0,:])
    N_spatial = len(_CSfield)

    # Spatial Fourier transform -- column-by-column i.e. each tau slice is FFT'd 
    for j in range(0, ntau):
      _CSfield[:,j] = fft(_CSfield[:, j])

    # Now do the Fourier transform in imaginary time to Matsubara freq.

    for m in range(0, N_spatial):
      _CSfield[m,:] = fft(_CSfield[m, :])

    return _CSfield


def ifft_dp1(_CSfield):
    # d+1 Fourier transform of a CS field object -- 2D numpy array \in C^ (Nx**dim , ntau)
    ntau = len(_CSfield[0,:])
    N_spatial = len(_CSfield)

    # Spatial Fourier transform -- column-by-column i.e. each tau slice is FFT'd 
    for j in range(0, ntau):
      #_CSfield[:,j] = ifft(_CSfield[:, j]) * N_spatial
      _CSfield[:,j] = ifft(_CSfield[:, j]) 
 
    # Now do the Fourier transform in imaginary time to Matsubara freq.

    for m in range(0, N_spatial):
      #_CSfield[m,:] = ifft(_CSfield[m, :]) * ntau
      _CSfield[m,:] = ifft(_CSfield[m, :])

    return _CSfield






# Parameters
ensemble = 'CANONICAL'
_mu = -1.0
_g = 0.000   # ideal gas if == 0 
ntau = 16
dim = 3
Nx = 1
L = 50  # simulation box size 
Vol = L**dim
_numspecies = 1

#N_input = 1000
N_input = 25 

N_spatial = Nx**dim

dV = Vol/N_spatial

#IC = np.ones((N_spatial, ntau), dtype=np.complex_) * (1/(np.sqrt(2)) + 1j*0) 
IC = np.zeros((N_spatial, ntau), dtype=np.complex_) 

# Need IC to be on the manifold 
IC += np.sqrt((N_input/ Vol))

beta = 0.5
lambda_psi = 0.001
_lambda = 6.0505834240

dt = 0.005
# Load the inputs

# inputs for gradient descent
 #with open('input.yml') as infile:
 #  inputs = yaml.load(infile, Loader=yaml.FullLoader)


numtsteps = 10
iofreq = 1   # print every 1000 steps 
#iofreq = 100 #  print every 1000 steps 

num_points = math.floor(numtsteps/iofreq)


# initialize psi at saddle pt
isPsizero = False 
_psi = 0. + 1j * (_mu) 

# initial value 
_lagrange_multiplier = 0.00

_CLnoise = True

print()
print()
print('-----Bosefluid Simulation: Bosonic Coherent States----')
print()
print()
print('Ensemble: ' + ensemble)
print()
if(ensemble == 'GRAND'):
    print('Chemical Potential: ' + str(_mu))
else:
    # Assmed CE (CANONICAL)
    print('N constraint: ' + str(N_input))
    print()
    print('lambda_psi Mobility: ' + str(lambda_psi))

print()
print()
print('Pair Repulsion Potential Strength: ' + str(_g))
print()
print('Temperature : ' + str(1/beta) + ' Kelvin')
print()
print('Running for ' + str(numtsteps) + ' timesteps')
print()
print('Using Ntau = ' + str(ntau) + ' tau slices' )
print()
print('Using Nx = ' + str(Nx) + ' grid points per dimension' )
print()
print('Using L = ' + str(L) + ' grid length per dimension' )
print()
print('Volume = ' + str(Vol))
print()
print('dV = ' + str(dV))
print()
print()
print('Complex Langevin Sampling')
print()
print(' Using timestep: ' + str(dt))
print()
print(' Using Noise? ' + str(_CLnoise))
print()
# initialize CS fields at zero
# d+1 dim fields: use 2D np arrays: (Nx **d) x Ntau 
phi = np.zeros((Nx**dim, ntau), dtype=np.complex_)
phistar = np.zeros((Nx**dim, ntau), dtype=np.complex_)
dSdphi = np.zeros((Nx**dim, ntau), dtype=np.complex_)
dSdphistar = np.zeros((Nx**dim, ntau), dtype=np.complex_)



# Fill with initial condition 
phi += IC
phistar += IC

# option to do normally distributed random nums
#phi += 1j*np.random.normal(0, 1.0, ntau) 
#phistar += 1j*np.random.normal(0, 1.0, ntau) 


dtau = beta/ntau


# Noise fields 
noise = np.zeros((Nx**dim, ntau), dtype=np.complex_)
noisestar = np.zeros((Nx**dim, ntau), dtype=np.complex_)

# Compute the linear and non-linear coefficients once since they are complex scalars and not a function of the configuration for a single spin in this model 
lincoef = np.zeros((Nx**dim, ntau), dtype=np.complex_)
nonlincoef = np.zeros((Nx**dim, ntau), dtype=np.complex_)
noisescl = np.zeros((Nx**dim, ntau), dtype=np.complex_)
nonlinforce = np.zeros((Nx**dim, ntau), dtype=np.complex_)


# Set up the spatial/k-grids
#x_grid = np.arange(0., L, L/Nx) 
 #kx_grid = np.linspace((-Nx/2 +1)*2.*np.pi/L , (Nx/2.)*2.*np.pi/L, Nx) 
 #kx_grid = np.sort(kx_grid) # optional sorting step 
 ##kx_grid = np.linspace((-Nx/2 +1)*np.pi/L , (Nx/2.)*np.pi/L, Nx) 
 #if(dim > 1):
 #  if(dim > 2):
 #    z_grid = x_grid 
 #    kz_grid = kx_grid 
 #  y_grid = x_grid # assumes cubic mesh  
 #  ky_grid = kx_grid 
 #
 #
 #X,Y,Z = np.meshgrid(x_grid, y_grid, z_grid)
 #
 #_k2_grid = np.zeros(Nx**dim)
 #KX, KY, KZ = np.meshgrid(kx_grid, ky_grid, kz_grid) 
 ## Fill k2-grid 
 ## Attempt 0: flatten
 #_k2_grid += (KX*KX + KY*KY + KZ*KZ).flatten()
 #
 ## attempt 1 -- for loops 
 # #for x in range(0, Nx): 
 # #  _k2_grid += np.sum(KX*KX)
 #
 ## attempt 2 - load in grid from csbosonscpp
 #k2data = np.loadtxt('k2map.dat', unpack=True)
 #_k2_grid = k2data[6] # 7th column is k^2 data  
_k2_grid = 0.0

print(_k2_grid)
# Sampling vectors   


t_s = np.zeros(num_points + 1)
N_tot_s = np.zeros(num_points + 1, dtype=np.complex_)
N2_s = np.zeros(num_points + 1, dtype=np.complex_)
psi_s = np.zeros(num_points + 1, dtype=np.complex_)
num_iters_s = np.zeros(num_points+1, dtype=np.complex_)


# initialize container for the density 
rho = np.zeros(Nx**dim, dtype=np.complex_)

# Calculate the particle numbers
N_tot = 0. 
for itau in range(0, int(ntau)):
  itaum1 = ( (int(itau) - 1) % int(ntau) + int(ntau)) % int(ntau)
  rho += phistar[:, itau] * phi[:, itaum1]
  
# scale by ntau
N = integrate_r_intensive(rho/ntau) * Vol

N2 = N**2 

psi_s[0] = _psi
N_tot_s[0] = N_tot
N2_s[0] = N2

# initialize the fictitious time 
t = 0.

# Prefill the linear coefficients 
for j in range(0, ntau):
  lincoef[:, j] = A_nk(j, ntau, beta, _k2_grid, _lambda, ensemble, _mu)
  # Correct diverging terms by using Euler limit of ETD
  # Python's FFT accounts for scaling, i.e. ifft(fft(a) == a , therefore, take out the scaling factors  
  for m in range(0, N_spatial): 
    if(lincoef[m, j] == 0.):
      nonlincoef[m, j] = -1. * ntau * dt 
      noisescl[m, j] = np.sqrt(ntau * dt / dV)
    else: 
      nonlincoef[m, j] = (np.exp(-lincoef[m,j] * ntau * dt) - 1.)/lincoef[m, j]
      noisescl[m, j] = np.sqrt((1. - np.exp(-2. * lincoef[m, j] * ntau * dt))/(2. * lincoef[m, j] * dV))
  lincoef[:, j] = np.exp(- lincoef[:, j] * ntau * dt)


N_tot_avg = 0. + 1j*0 
N2_avg = 0. + 1j*0

ctr = 1


# Iteration vars 
num_iters_s = np.zeros(num_points + 1)
num_iters_s[0] = 0

# Initialize Matrices  
num_DOF = 2 * ntau
I = np.identity(num_DOF, dtype=np.complex_)  # 1 species  
state_vector = np.zeros(num_DOF, dtype=np.complex_)

G_matrix = np.zeros(((num_DOF) + 1, (num_DOF)+ 1), dtype=np.complex_)  # 2ntau DOF + 1 constraint DOF for single species Bosefluid  
d_vector = np.zeros(num_DOF + _numspecies, dtype=np.complex_) 

y_iter = np.zeros(num_DOF, dtype=np.complex_) 


# Fill G_matrix identity part
for i in range(0, num_DOF):
  G_matrix[i,i] = 1.


# grad e
grad_e = np.zeros(len(y_iter), dtype=np.complex_)
y_tilde = np.zeros(len(y_iter), dtype=np.complex_) 
y_ETD = np.zeros(len(y_iter), dtype=np.complex_) 
y_vector = np.zeros(len(y_iter), dtype=np.complex_) 
y1 = np.zeros(len(y_iter), dtype=np.complex_)
all_noises= np.zeros(2*ntau, dtype=np.complex_)


# Initialize y_vector with the CS fields 
y_vector += np.hstack([phi[0], phistar[0]])
max_iterations = 10
ETOL = 1E-2
num_iters_avg = 1

_isProjection = True

start = time.time()

_psi = 0 + 1j*0
_psi += _lagrange_multiplier
# Timestep using ETD 
for l in range(0, numtsteps + 1):
#for l in range(0, 2):
  if(_isProjection):
    fill_forces(phi, phistar, dSdphistar, dSdphi, ntau, -0.1j, ensemble, _g, beta)
    fill_grad_e(phi, phistar, grad_e)
  #Forces.fill(0.)
  #Forces += np.hstack([dSdphi, dSdphistar]) # Diagonal relaxation 
  #Forces += np.hstack([dSdphistar, dSdphi]) # off-diagonal relaxation  

  # Projection method:
  # Step 1, hop off manifold
  # Step 2, perform ETD step w/o lagrange multiplier
  # Step 3, hop back on manifold using same lagrange multiplier as step 1) 

  # Converge steps 1 - 3 using a Newton iteration 

  #print(dSdphi)
  # Generate Noise terms for the whole time step 
  noise.fill(0.) 
  noisestar.fill(0.) 
  noise = np.random.normal(0, 1., (N_spatial, ntau)) + 1j * np.random.normal(0, 1., (N_spatial, ntau))
  noisestar = np.conj(noise) 
  
  # FFT and Scale by fourier coeff 
  noise = fft_dp1(noise) * noisescl 
  noisestar = fft_dp1(noisestar) * np.conj(noisescl) 

  # already scaled and in k, tau rep.
  #all_noises.fill(0.) 
  #all_noises += np.hstack([noise[0], noisestar[0]]) 
  #all_noises *= np.sqrt(2. * mobility * dt)


  # Set up Newton's iteration
  # Use previous psi as initial iterate guess for psi  
  psi_iter = 0. + 1j * 0.
  psi_iter += _psi 
  #psi_iter += _psi 
  # initial iterate/guess: phi_vector EM stepped with psi term (typical CL), no noise
  y_iter.fill(0.) # reset to zero 

  # Intial iteration guess -- an EM step
  # Try ETD step, no noise 
  if(_isProjection):
    phi, phistar = ETD(phi, phistar, dSdphistar, dSdphi, lincoef, nonlincoef, np.zeros(ntau, dtype=np.complex_), np.zeros(ntau, dtype=np.complex_))
  else:
    fill_forces(phi, phistar, dSdphistar, dSdphi, ntau, _psi, ensemble, _g, beta)
    phi, phistar = ETD(phi, phistar, dSdphistar, dSdphi, lincoef, nonlincoef, noise, noisestar) 

  #print('residual pre iteration:', str(constraint_err(N_input, phi, phistar)))
  y_iter += np.hstack([phi[0], phistar[0]]) 

  #print('printing y_iter')
  #print(y_iter)
  #y_iter -= mobility * dt * Forces # EM 
  if(_isProjection):
    for j in range(0, max_iterations + 1):
      # Do the Naive Euler step
      # calculate y_tilde 
      y_tilde.fill(0.)
      y_tilde += y_vector 
  
      phi.fill(0.) 
      phistar.fill(0.) 
      phi[0], phistar[0] = np.split(y_vector, 2) # use the original phi,phi* for gradE
  
      fill_grad_e(phi, phistar, grad_e)
      #print('psi iter: ', psi_iter)
      y_tilde += grad_e * psi_iter 
      #y_tilde += grad_e * psi_iter * -1j   # current psi iteration 
     
      # Do Euler maruyama / ETD on y_tilde
      phi.fill(0.) 
      phistar.fill(0.) 
      phi[0], phistar[0] = np.split(y_tilde, 2)
      fill_forces(phi, phistar, dSdphistar, dSdphi, ntau, 0., ensemble, _g, beta)
      #Forces = np.hstack([dSdphistar, dSdphi]) 
      #y_EM = y_tilde - mobility * dt * Forces  + all_noises
      #print(phi) 
      if(_CLnoise):
        phi, phistar = ETD(phi, phistar, dSdphistar, dSdphi, lincoef, nonlincoef, noise, noisestar)
      else:
        phi, phistar = ETD(phi, phistar, dSdphistar, dSdphi, lincoef, nonlincoef, np.zeros(ntau, dtype=np.complex_), np.zeros(ntau, dtype=np.complex_))
  
      #print(phi) 
      y_ETD.fill(0.)
      y_ETD += np.hstack([phi[0], phistar[0]])
      
      # reset size of y1 vector 
      # fill it with the next iteration  
      y1.fill(0.) # zero 
      y1 += y_iter
  
      #print('printing y1') 
      #print(y1)
      # Evalute grad e at y1  
      phi.fill(0.) 
      phistar.fill(0.) 
      phi[0], phistar[0] = np.split(y1, 2)
      fill_grad_e(phi, phistar, grad_e); # filling gradE at y_l+1 
  
      #print(grad_e)
      #print('psi iter: ', psi_iter)
      y1 -= (grad_e * psi_iter)
      # Take EM step 
      y1 -= y_ETD   # residuals 
  
      # Refill the G matrix 
      phi.fill(0.) 
      phistar.fill(0.) 
      phi[0], phistar[0] = np.split(y_iter, 2)
  
      # fill constraint vector 
      d_vector.fill(0.) 
      d_vector[-1] += constraint_err(N_input, phi, phistar)  # last entry
      d_vector[0:num_DOF] += y1
      d_vector *= -1. # "b" vector to do Ax = b
  
      phi[0], phistar[0] = np.split(y_iter, 2)
      fill_grad_e(phi, phistar, grad_e);
      #print('printing grad e')
      #print(grad_e)
      
      for i in range(0, num_DOF): 
        G_matrix[i,-1] = -2. # final column 
        G_matrix[i,-1] *= grad_e[i] 
        G_matrix[-1,i] = 1.   # final row 
        G_matrix[-1,i] *= grad_e[i] 
  
      G_matrix[-1,-1] = 0. 
   #
   #    if l == 1:
   #      print('printing G')
   #      print(G_matrix)
  
  
      #print('printing d_vector')
      #print(d_vector)
      
      # do Ax = b, using G as A and delta as b
      solution = np.linalg.solve(G_matrix, d_vector)
      print('solution: ', solution) 
  
      # Update iteration with the solution (updates) 
      y_iter += solution[0:num_DOF] 
      psi_iter += solution[-1]
  
      #if(np.all(solution < ETOL)):
      if(np.abs(d_vector[-1]) < ETOL and np.all(solution < ETOL)):
        num_iters_avg += j+1/iofreq
        num_iters = j + 1
        print('completed a timestep')
        print('constraint residual: ' + str(d_vector[-1]))
        print('psi value: ' + str(psi_iter))
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
    phi.fill(0.) 
    phistar.fill(0.) 
    phi[0], phistar[0] = np.split(y_vector, 2)




  # ----- Concluded full step/projection iteration -----  
  # ----- Next, calculate observables ----- 


  # Calculate the particle numbers
  N_tot = 0.
  rho.fill(0.) 
  for itau in range(0, ntau):
    itaum1 = ( (int(itau) - 1) % int(ntau) + int(ntau)) % int(ntau)
    rho += phistar[:, itau] * phi[:, itaum1]
  
  rho *= 1./float(ntau)
  N_tot = integrate_r_intensive(rho) * Vol 
  N2 = N_tot**2 

 
  if(np.isnan(N2)):
    print('Trajectory diverged at iteration: ' + str(l) + ' and CL time = ' + str(t))
    break

  # Step the psi field
  if(not _isProjection):
    if(ensemble == "CANONICAL"):
      _psi -= 1j * (lambda_psi * dt) * (N_input - N_tot)
      # Add the psi noise
      if(_CLnoise): 
        psi_noisescl = np.sqrt(2. * lambda_psi * dt) 
        eta = np.random.normal() * psi_noisescl 
        _psi += eta 

 
  # Calculate observables - sample   
  N_tot_avg += N_tot/iofreq 
  N2_avg += N2/iofreq

  t += dt

  # Output on interval
  if(l % iofreq == 0 and l > 0):
     if(ctr %  25):
       print("Completed {} of {} steps".format(l, numtsteps))
     # opout.write("{} {} {} {} {}\n".format(it, Msum.real / Navg, Msum.imag / Navg, psi.real, psi.imag))
     t_s[ctr] = t 
     N_tot_s[ctr] = N_tot_avg 
     num_iters_s[ctr] = num_iters_avg 
     N2_s[ctr] = N2_avg
    
     if(ensemble == "CANONICAL"):
       psi_s[ctr] = _psi
     # clear the averages 
     N2_avg = 0. + 1j*0 
     N_tot_avg = 0. + 1j*0 
     num_iters_avg = 0.
     ctr += 1

    

end = time.time()
print()
print()
print('Simulation finished: Runtime = ' + str(end - start) + ' seconds')




# Print the results (noise long-time averages)
print()
print()
print('The Boson Particle Number is: ' + str(np.mean(N_tot_s[10:].real)))
print()
print('The Particle Number squared is: ' + str(np.mean(N2_s[10:].real)))
print()
print('The density is: ' + str(np.mean(N_tot_s[10:].real)/Vol))
print()

# plot the results 

plt.figure(1)
plt.title('Particle Number: CL Simulation', fontsize = 20, fontweight = 'bold')
plt.plot(t_s, N_tot_s.real, '*-', color = 'green', linewidth = 0.5, label = 'Samples: real')
plt.plot(t_s, N_tot_s.imag, '*-', color = 'skyblue', linewidth=0.5,label = 'Samples: imag')
#plt.plot(t_s, np.ones(len(t_s)), 'k', label = 'Constraint')
plt.xlabel('CL time', fontsize = 20, fontweight = 'bold')
plt.ylabel('$N_{tot}$', fontsize = 20, fontweight = 'bold') 
#plt.ylim([-5, 40])
plt.legend()
plt.show()


if(ensemble == "CANONICAL"):
    _mu_CL = 1j*psi_s/beta
    print()
    print('The average chemical potential is: ' + str(np.mean(_mu_CL[10:].real)))
    print()
    plt.figure(2)
    plt.title('Psi or $\mu$ sampling: CL Simulation', fontsize = 20, fontweight = 'bold')
    plt.plot(t_s, _mu_CL.real, '*-r', label = 'Samples: real')
    plt.plot(t_s, _mu_CL.imag, '*-g', label = 'Samples: imag')
    plt.xlabel('CL time', fontsize = 20, fontweight = 'bold')
    plt.ylabel('$\psi$', fontsize = 20, fontweight = 'bold') 
    #plt.ylim([-5, 5])
    plt.legend()
    plt.show()

plt.figure(8)
plt.title('Iterations per timestep sampling: CL Simulation', fontsize = 20, fontweight = 'bold')
plt.plot(t_s, num_iters_s, color = 'red', marker='o', label = 'Projected CL')
plt.xlabel('CL time', fontsize = 20, fontweight = 'bold')
plt.ylabel('Iterations', fontsize = 20, fontweight = 'bold') 
#plt.ylim([0, 10])
plt.legend()
plt.show()
    
 #    plt.figure(3)
 #    plt.title('Psi Trajectory: CL Simulation', fontsize = 20, fontweight = 'bold')
 #    plt.plot(_mu_CL.real, _mu_CL.imag, '*-r', label = 'Samples: real')
 #    plt.xlabel('Re($\psi$)', fontsize = 20, fontweight = 'bold')
 #    plt.ylabel('Im($\psi$)', fontsize = 20, fontweight = 'bold') 
 #    plt.legend()
 #    # plt.ylim([-2, 1])
 #    plt.show()



plt.figure(7)
plt.title('$N^2$ : CL Simulation', fontsize = 20, fontweight = 'bold')
plt.plot(t_s, N2_s.real, '*-', color = 'purple', label = 'Samples')
plt.plot(t_s, N2_s.imag, '*-', color = 'skyblue', label = 'Samples')
# plt.plot(t_s, np.ones(len(t_s)), 'k', label = 'Constraint')
plt.xlabel('CL time', fontsize = 20, fontweight = 'bold')
plt.ylabel('$N^2$', fontsize = 20, fontweight = 'bold') 
#plt.ylim([-5, 20])
plt.legend()
plt.show()

