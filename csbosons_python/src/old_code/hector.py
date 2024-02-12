import numpy as np
import yaml
import math
import matplotlib
#matplotlib.rcParams['text.usetex'] = True
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt 
import time
from scipy.fft import fft 
from scipy.fft import ifft
import matplotlib 
import matplotlib.pyplot as plt 


# Python function for executing CL sampling of a bosefluid in the coherent states field theoretic representation 
# Helper functions


# Refresh linear force coefficient
def A_nk(n, _ntau, _beta, k2_grid, _lambda, _ensemble, _mu):
        # Returns a d-dimensional field, evaluated at timeslice "n" 
        A = np.zeros(len(k2_grid), dtype=np.complex_) 
        dtau = _beta/_ntau
        A += (- dtau * _lambda * k2_grid) # hbar^2/2m k^2, kinetic energy 
        if(_ensemble == "GRAND"):
          A += (_mu * dtau)
        A += 1. 
        A *= -np.exp(-2. * np.pi * 1j * n / _ntau) 
        A += 1.
        return A 


def fill_forces(phi, phistar, dSdphistar, dSdphi, ntau, _psi, _ensemble, _g, _beta, _coeff_phi, _coeff_phistar, _isShifting, A_n):
  # Fills the nonlinear forces as applicable  
  dSdphistar.fill(0.)
  dSdphi.fill(0.)
  #_coeff_phi.fill(0.) 
  #_coeff_phistar.fill(0.) 

  dtau = _beta/ntau
  for itau in range(0, int(ntau)):
    # PBC 
    itaum1 = ( (int(itau) - 1) % int(ntau) + int(ntau)) % int(ntau)
    # compute avg. rho
    avg_rho = 0.
    #avg_rho += phi[:,itaum1] * phistar[:, itau] / ntau
    avg_rho = 1.1

 #    # If we are doing diagonal force stepping, we must include the linear parts that were omitted 
 #    if(not _isOffDiagonal):
 #      dSdphistar[itau] += A_nk(itau, ntau, _beta, k2_grid, _lambda 
    # Build force vector
    # nonlinear forces  
    dSdphistar[:, itau] += _g * dtau * phi[:, itaum1] * phi[:, itaum1] * phistar[:, itau]
    dSdphi[:, itaum1] += _g * dtau * phistar[:, itau] * phi[:, itaum1] * phistar[:, itau]

    # compute the coefficient for phi and phistar 
    if(_isShifting):
      _coeff_phi[:, itau] = avg_rho * dtau * _g * itau * np.exp(-2. * np.pi * 1j * itau / ntau) # this performs best 
      #_coeff_phi[:, itau] = avg_rho * dtau * _g * itau * np.exp(-2. * np.pi * 1j * itau / ntau) # this performs best 

      # 1. shift to ensure real, pos. definite 
      if((_coeff_phi[:, itau] + A_n[:,itau]).real < 0.):
        #print('coef is not positive')
        #print('mode: ' + str(itau))
        _coeff_phi[:, itau] -= 2.*(_coeff_phi[:, itau] + A_n[:, itau]).real # reflect onto real axis  
        #if((_coeff_phi[:, itau] + A_n[:,itau]).real > 0.):
        #  print('coef is now real')
      # 2. shift to ensure pure real in linear coeff.
      _coeff_phi[:, itau] -= (_coeff_phi[:, itau] + A_n[:, itau]).imag

      #_coeff_phi[:, itau] = avg_rho * dtau * _g * itau * np.exp(-2. * np.pi * 1j * itau / ntau)  + itau * itau # this performs best 
      #_coeff_phi[:, itau] = avg_rho * dtau * _g * itau * np.exp(-2. * np.pi * 1j * itau / ntau)  
      #_coeff_phi[:, itau] = (itau) ** 2
      #_coeff_phi[:, itau] = (itau * np.pi * 2. / _beta) ** 2
      _coeff_phistar[:, itau] = np.conj(_coeff_phi[:, itau]) 
      #_coeff_phistar[:, itau] = avg_rho * dtau * _g * np.exp(2. * np.pi * 1j * itau / ntau)
    
    if(_ensemble == "CANONICAL"):
      # include mu force in nonlinear
      # identify _mu = i psi / \beta
      _mu_eff = 1j * _psi / _beta
      dSdphistar[:, itau] += -phi[:, itaum1] * _mu_eff * dtau 
      dSdphi[:, itaum1] +=  -phistar[:, itau] * _mu_eff * dtau 

  if(_isShifting): 
    # FFT CS fields and forces 
    phi = fft_dp1(phi)
    phistar = fft_dp1(phistar)
    dSdphistar = fft_dp1(dSdphistar)
    dSdphi = fft_dp1(dSdphi)
    _dbg = False
    #_coeff_phi += dSdphistar / phi
    #_coeff_phistar += dSdphi / phistar
    if(_dbg):
      #elem = 4
      print()
 #      print('force')
 #      print(dSdphi[0,elem])
 #      print('approx')
 #      print(_coeff_phi[0,elem] * phi[0,elem])
 #      print('difference')
 #      print(dSdphi[0,elem] - (_coeff_phi[0,elem] * phi[0,elem]))
      print()
      print(np.max(np.abs(dSdphi[0,:] - (_coeff_phi[0,:] * phi[0,:]))))

    dSdphistar -= (_coeff_phi * phi) # must be in matsubara representation
    dSdphi -= (_coeff_phistar * phistar)
    # iFFT back; rest of the code expects forces and phi in r, tau space 
    phi = ifft_dp1(phi)
    phistar = ifft_dp1(phistar)
    dSdphistar = ifft_dp1(dSdphistar)
    dSdphi = ifft_dp1(dSdphi)


def fill_grad_e(phi, phistar,  grad_e): 
  L.fill(0.)
  Lstar.fill(0.)


  # Perform index shifts to get the gradient constraint vectors 
  for itau in range(0, ntau):
    # PBC 
    itaum1 = ( (int(itau) - 1) % int(ntau) + int(ntau)) % int(ntau)
    L[:, itau] += phi[:, itaum1]
    Lstar[:, itaum1] += phistar[:, itau]
  
  grad_e.fill(0.)
  grad_e += np.hstack([Lstar, L])
  # grad_e += np.hstack([L_up, Lstar_up, L_dwn, Lstar_dwn]) # off-diagonal relaxation
  grad_e *= 1./float(ntau)



def integrate_r_intensive(field):
    N_spatial = len(field)
    result = np.sum(field)/N_spatial
    return result




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



def constraint_err(_N_input, phi, phistar):
    # Function for calculation the constraint error
    N_spatial = len(phi)
    tmp = np.zeros(N_spatial, dtype=np.complex_)
    Ntau = len(phi[0, :]) 
    for itau in range(0, Ntau):
      itaum1 = ( (int(itau) - 1) % int(Ntau) + int(Ntau)) % int(Ntau)
      tmp += phistar[:, itau] * phi[:, itaum1]
    tmp *= 1./Ntau

    constraint_residual = integrate_r_intensive(tmp) * Vol
    constraint_residual -= _N_input
    return constraint_residual  # should be near zero 



def ETD(phi, phistar, _dSdphistar, _dSdphi, _lincoef, _lincoef_phistar, _nonlincoef, _nonlincoef_phistar, noisescl, noisescl_phistar, _CLnoise):
    ntau = len(phi[0, :])
    N_spatial = len(phi)
    # Exponential-Time-Differencing, assumes off-diagonal stepping 

    # Function to step phi and phistar with ETD  
    phi = fft_dp1(phi) * (_lincoef) 
    phistar = fft_dp1(phistar) * (_lincoef_phistar) 

    # add nonlinear term, off-diagonal relaxation 
    phi += (fft_dp1(_dSdphistar) * _nonlincoef)
    phistar += (fft_dp1(_dSdphi) * _nonlincoef_phistar)

    # noise
    _noise = np.zeros((N_spatial, ntau), dtype=np.complex_)
    _noisestar = np.zeros((N_spatial, ntau), dtype=np.complex_)
    _noise.fill(0.) 
    _noisestar.fill(0.) 

    if(_CLnoise):
      # ETD assumes off-diagonal stepping, generate nosie and scale  
      _noise += np.random.normal(0, 1., (N_spatial, ntau)) + 1j * np.random.normal(0, 1., (N_spatial, ntau))
      _noisestar += np.conjugate(_noise) 
      # FFT and Scale by fourier coeff 
      _noise = fft_dp1(_noise) * noisescl 
      #_noisestar = fft_dp1(_noisestar) * np.conj(noisescl) 
      _noisestar = fft_dp1(_noisestar) * noisescl_phistar 
      # Add the noise to CS fields  
      phi += _noise 
      phistar += _noisestar 

    # inverse fft  
    phi = ifft_dp1(phi) 
    phistar = ifft_dp1(phistar) 

    return [phi, phistar]
    # Return state vector (packaged phi/phistar vector)



def EM_implicit(phi, phistar, dSdphistar, dSdphi, _isOffDiagonal, _CLnoise, dV, dt, linearcoeff, linearcoeff_star):
    _tolerance = 1E-14
    max_iters = 100
    num_iters = 1
    cost = 0.1
    ntau = len(phi[0, :])
    N_spatial = len(phi)

    phi_cp = np.zeros((Nx**dim, ntau), dtype=np.complex_)
    phistar_cp = np.zeros((Nx**dim, ntau), dtype=np.complex_)
    _tmp = np.zeros((Nx**dim, ntau), dtype=np.complex_)
    _tmp2 = np.zeros((Nx**dim, ntau), dtype=np.complex_)

    phi_cp += phi
    phistar_cp += phistar

    # noise
    noise = np.zeros((N_spatial, ntau), dtype=np.complex_)
    noisestar = np.zeros((N_spatial, ntau), dtype=np.complex_)
    noise.fill(0.) 
    noisestar.fill(0.) 
    mobility = ntau
    #mobility = 1. 
    noisescl_scalar = np.sqrt(mobility * dt) 
    #noisescl_scalar = np.sqrt(mobility * dt / dV)

    # Do an initial EM step 
    phi -= dSdphistar * mobility * dt 
    phistar -= dSdphi * mobility * dt
    if(_CLnoise): 
      noise = np.random.normal(0, 1., (N_spatial, ntau)) + 1j * np.random.normal(0, 1., (N_spatial, ntau))
      noisestar = np.conj(noise)
      noise *= noisescl_scalar
      noisestar *= noisescl_scalar
      phi += noise 
      phistar += noisestar 

    while(cost > _tolerance):
      # Calculate forces with new CS fields 
      # Update the nonlinear forces before stepping; fill forces clears dSdphistar and dSdphi 
      fill_forces(phi, phistar, dSdphistar, dSdphi, ntau, _psi, ensemble, _g, beta, _coeff_phi, _coeff_phistar, False)

      # recompute total forces 
      # d+1 FFT force container
      dSdphistar = fft_dp1(dSdphistar) 
      dSdphi = fft_dp1(dSdphi) 
      # Add linearized contributions 
      dSdphistar += linearcoeff * fft_dp1(phi)
      dSdphi += linearcoeff_star * fft_dp1(phistar)
  
      # need to iFFT phi and phistar
      phi = ifft_dp1(phi) 
      phistar = ifft_dp1(phistar)
  
      # inverse d+1 FFT force container
      dSdphistar = ifft_dp1(dSdphistar) 
      dSdphi = ifft_dp1(dSdphi) 

      _tmp.fill(0.)
      _tmp2.fill(0.)
      _tmp += phi
      _tmp2 += phistar
      # Reset fields to initial state 
      phi.fill(0.)
      phistar.fill(0.)
      phi += phi_cp
      phistar += phistar_cp

      # Do an EM step, using forces evaluated at phi^{l+1}, starting w. fields at l 
      phi -= dSdphistar * mobility * dt 
      phistar -= dSdphi * mobility * dt
      # Add the noise 
      if(_CLnoise): 
        phi += noise 
        phistar += noisestar 

      # prep for cost 
      _tmp -= phi 
      _tmp2 -= phistar 

      cost = 0.
      cost = np.max(np.abs(_tmp)) + np.max(np.abs(_tmp2))
      num_iters += 1

      #print(cost)
      #print(num_iters)
      if(cost < _tolerance):
        #print(num_iters)
        #print(cost)
        break

      if(num_iters > max_iters):
        print('Warning, we have exceeded the max number of iterations!')
        break

    return [phi, phistar]
    # Return state vector (packaged phi/phistar vector)

def ETD_implicit(phi, phistar, dSdphistar, dSdphi, lincoef, lincoef_phistar, nonlincoef, nonlincoef_phistar, noisescl, noisescl_phistar, _CLnoise):

    _tolerance = 1E-10
    max_iters = 100
    num_iters = 1
    cost = 0.1
    ntau = len(phi[0, :])
    N_spatial = len(phi)

    phi_cp = np.zeros((Nx**dim, ntau), dtype=np.complex_)
    phistar_cp = np.zeros((Nx**dim, ntau), dtype=np.complex_)
    tmp = np.zeros((Nx**dim, ntau), dtype=np.complex_)
    tmp2 = np.zeros((Nx**dim, ntau), dtype=np.complex_)

    phi_cp += phi
    phistar_cp += phistar

    # Need to generate noise and fix it throughout iterations 
    # noise
    noise = np.zeros((N_spatial, ntau), dtype=np.complex_)
    noisestar = np.zeros((N_spatial, ntau), dtype=np.complex_)

    noise.fill(0.) 
    noisestar.fill(0.) 

    if(_CLnoise):
      # ETD assumes off-diagonal stepping, generate nosie and scale  
      noise = np.random.normal(0, 1., (N_spatial, ntau)) + 1j * np.random.normal(0, 1., (N_spatial, ntau))
      noisestar = np.conj(noise) 
      # FFT and Scale by fourier coeff 
      noise = fft_dp1(noise) * noisescl 
      #noisestar = fft_dp1(noisestar) * np.conj(noisescl) 
      noisestar = fft_dp1(noisestar) * noisescl_phistar 

    # do ETD step to start 
    phi, phistar = ETD(phi, phistar, dSdphistar, dSdphi, lincoef, lincoef_phistar, nonlincoef, nonlincoef_phistar, noisescl, noisescl_phistar, False)
    # Add the noise 
    if(_CLnoise):
      phi = fft_dp1(phi) 
      phistar = fft_dp1(phistar) 
      phi += noise 
      phistar += noisestar
      phi = ifft_dp1(phi) 
      phistar = ifft_dp1(phistar) 

    while(cost > _tolerance):
      # Calculate forces with new CS fields 
      # Update the nonlinear forces before stepping; fill forces clears dSdphistar and dSdphi 
      fill_forces(phi, phistar, dSdphistar, dSdphi, ntau, _psi, ensemble, _g, beta, _coeff_phi, _coeff_phistar, False)

      tmp.fill(0.)
      tmp2.fill(0.)
      tmp += phi
      tmp2 += phistar
      # Reset fields to initial state 
      phi.fill(0.)
      phistar.fill(0.)
      phi += phi_cp
      phistar += phistar_cp

      # step the fields 
      phi, phistar = ETD(phi, phistar, dSdphistar, dSdphi, lincoef, lincoef_phistar, nonlincoef, nonlincoef_phistar, noisescl, noisescl_phistar, False)
      # Add the noise 
      if(_CLnoise):
        phi = fft_dp1(phi) 
        phistar = fft_dp1(phistar) 
        phi += noise 
        phistar += noisestar 
        phi = ifft_dp1(phi) 
        phistar = ifft_dp1(phistar) 
      # prep for cost 
      tmp -= phi 
      tmp2 -= phistar 

      cost = 0.
      cost = np.max(np.abs(tmp)) + np.max(np.abs(tmp2))
      num_iters += 1

      #print(cost)
      #print(num_iters)
      if(cost < _tolerance):
        #print(num_iters)
        break

      if(num_iters > max_iters):
        print('Warning, we have exceeded the max number of iterations!')
        break

    return [phi, phistar]
    # Return state vector (packaged phi/phistar vector)




def EM(phi, phistar, dSdphistar, dSdphi, _isOffDiagonal, _CLnoise, dV, dt):
    # Function to step phi and phistar with ETD  
    ntau = len(phi[0, :])
    N_spatial = len(phi[:, 0])

    # noise
    noise = np.zeros((N_spatial, ntau), dtype=np.complex_)
    noisestar = np.zeros((N_spatial, ntau), dtype=np.complex_)
    noise.fill(0.) 
    noisestar.fill(0.) 
    mobility = ntau
    #mobility = 1. 
    noisescl_scalar = np.sqrt(mobility * dt) 
    #noisescl_scalar = np.sqrt(mobility * dt / dV)

    if(_isOffDiagonal):
      phi -= dSdphistar * mobility * dt 
      phistar -= dSdphi * mobility * dt
      if(_CLnoise): 
        noise = np.random.normal(0, 1., (N_spatial, ntau)) + 1j * np.random.normal(0, 1., (N_spatial, ntau))
        noisestar = np.conj(noise)
    else:
      noisescl_scalar *= np.sqrt(2.) # Real noise, FDT 
      phi -= dSdphi * mobility * dt 
      phistar -= dSdphistar * mobility * dt
      # For diagonal stepping, generate real noise  
      if(_CLnoise): 
        noise = np.random.normal(0, 1., (N_spatial, ntau))  
        noisestar = np.random.normal(0, 1., (N_spatial, ntau)) 


    # Scale and Add the noise 
    if(_CLnoise): 
      noise *= noisescl_scalar
      noisestar *= noisescl_scalar

      phi += noise 
      phistar += noisestar 

    return [phi, phistar]




# Parameters
ensemble = 'GRAND'
_mu = 0.10
_g = 1.0      # ideal gas if == 0 
ntau = 64
dim = 1
Nx = 1
L = 1  # simulation box size 
beta = 1.00
Vol = L**dim

if(_mu < 0):
  N_input = 25.0 # ideal gas 
else:
  N_input = (_mu / _g * Vol) # particle number input 

N_spatial = Nx**dim

dV = Vol/N_spatial

#IC = np.ones((N_spatial, ntau), dtype=np.complex_) * (1/(np.sqrt(2)) + 1j*0) 
IC = np.zeros((N_spatial, ntau), dtype=np.complex_)
if _g == 0:
  IC += np.sqrt(N_input/Vol) 
else:
  IC += np.sqrt(_mu/_g) # homogeneous initial condition  
  #IC += np.sqrt(2.37323)

np.random.seed(1)
lambda_psi = 0.005
_lambda = 6.0505834240

dt = 0.01
#dt = 0.0025
#dt = 0.015
# Load the inputs

# inputs for gradient descent
 #with open('input.yml') as infile:
 #  inputs = yaml.load(infile, Loader=yaml.FullLoader)


numtsteps = 50000
iofreq = 200  # print every 1000 steps 
#iofreq = 100 #  print every 1000 steps 

num_points = math.floor(numtsteps/iofreq)


# initialize psi at saddle pt
isPsizero = False 
_psi = 0. + 1j * (_mu) 

_CLnoise = True

_isOffDiagonal = True
_ETD = False
_do_implicit = False
_isShifting = True


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

_coeff_phi = np.zeros((Nx**dim, ntau), dtype=np.complex_)
_coeff_phistar = np.zeros((Nx**dim, ntau), dtype=np.complex_)
# Fill with initial condition 
phi += IC
phistar += IC

# option to do normally distributed random nums
 #phi = np.random.normal(0, 1.0, ntau) 
 #phistar = np.random.normal(0, 1.0, ntau) 


dtau = beta/ntau


# Noise fields 
 #noise = np.zeros((Nx**dim, ntau), dtype=np.complex_)
 #noisestar = np.zeros((Nx**dim, ntau), dtype=np.complex_)


# Compute the linear and non-linear coefficients once since they are complex scalars and not a function of the configuration for a single spin in this model 
lincoef = np.zeros((Nx**dim, ntau), dtype=np.complex_)
lincoef_phistar = np.zeros((Nx**dim, ntau), dtype=np.complex_)
nonlincoef = np.zeros((Nx**dim, ntau), dtype=np.complex_)
nonlincoef_phistar = np.zeros((Nx**dim, ntau), dtype=np.complex_)
noisescl = np.zeros((Nx**dim, ntau), dtype=np.complex_)
noisescl_phistar = np.zeros((Nx**dim, ntau), dtype=np.complex_)
nonlinforce = np.zeros((Nx**dim, ntau), dtype=np.complex_)


# Set up the spatial/k-grids
#assert(Nx > 3)
if( Nx > 3):
  n_grid = np.append( np.arange(0, Nx/2 + 1, 1) , np.arange(-Nx/2 + 1, 0, 1) ) # artifical +1 required for 2nd argument 
elif( Nx == 1):
  n_grid = np.array([0]) 
dk = np.pi * 2 / L

x_grid = np.arange(0., L, L/Nx) 
kx_grid = n_grid * dk
assert(len(x_grid) == len(kx_grid))
#kx_grid = np.linspace((-Nx/2 + 1)*2.*np.pi/L , (Nx/2.)*2.*np.pi/L, Nx) 
#kx_grid = np.sort(kx_grid) # optional sorting step 
#kx_grid = np.linspace((-Nx/2 +1)*np.pi/L , (Nx/2.)*np.pi/L, Nx) 
if(dim > 1):
  if(dim > 2):
    z_grid = x_grid 
    kz_grid = kx_grid 
  y_grid = x_grid # assumes cubic mesh  
  ky_grid = kx_grid 


# these are not synced up with k-grid 
if(dim > 1):
  X,Y = np.meshgrid(x_grid, y_grid)
  if(dim > 2):
    X,Y,Z = np.meshgrid(x_grid, y_grid, z_grid)
else:
  X = x_grid

_k2_grid = np.zeros(Nx**dim)
_k2_grid_v2 = np.zeros(Nx**dim)
if(dim > 1):
  KX, KY = np.meshgrid(kx_grid, ky_grid) 
  _k2_grid_v2 += (KX*KX + KY*KY).flatten()
  if(dim > 2):
    KX, KY, KZ = np.meshgrid(kx_grid, ky_grid, kz_grid) 
    _k2_grid_v2 += (KX*KX + KY*KY + KZ*KZ).flatten()
else:
  KX = kx_grid 
  _k2_grid_v2 += (KX*KX).flatten() 

#k2data = np.loadtxt('k2map.dat', unpack=True)
_k2_grid += _k2_grid_v2

print(_k2_grid)



t_s = np.zeros(num_points + 1)
N_tot_s = np.zeros(num_points + 1, dtype=np.complex_)
N2_s = np.zeros(num_points + 1, dtype=np.complex_)
psi_s = np.zeros(num_points + 1, dtype=np.complex_)

# initialize container for the density 
rho = np.zeros(Nx**dim, dtype=np.complex_)

# Calculate the particle numbers
N_tot = 0. 
for itau in range(0, int(ntau)):
  itaum1 = ( (int(itau) - 1) % int(ntau) + int(ntau)) % int(ntau)
  rho += phistar[:, itau] * phi[:, itaum1]
  
# scale by ntau
N = integrate_r_intensive(rho/ntau) * Vol

print('initial particle number: ' + str(N))
print()
print('initial (average) density : ' + str(N/Vol))

N2 = N**2 

psi_s[0] = _psi
N_tot_s[0] = N 
N2_s[0] = N2

# initialize the fictitious time 
t = 0.

# Prefill the linear coefficients 
avg_rho = np.ceil( _mu/_g )
#avg_rho = 2.10 
_coeff_phi.fill(0.)
_coeff_phistar.fill(0.)
if(_isShifting):
  for itau in range(0, ntau):
    #_coeff_phi[:, itau] = avg_rho * dtau * _g * itau * np.exp(-2. * np.pi * 1j * itau / ntau) # this performs best 
    _coeff_phi[:, itau] = avg_rho * dtau * _g * itau * np.exp(-2. * np.pi * 1j * itau / ntau) # this performs best 
    #_coeff_phi[:, itau] = avg_rho * dtau * _g * itau * np.exp(-2. * np.pi * 1j * itau / ntau)  + itau * itau # this performs best 
    #_coeff_phi[:, itau] = avg_rho * dtau * _g * itau * np.exp(-2. * np.pi * 1j * itau / ntau)  
    #_coeff_phi[:, itau] += (itau) ** 2
    #_coeff_phi[:, itau] = (itau * np.pi * 2. / _beta) ** 2
    _coeff_phistar[:, itau] = np.conj(_coeff_phi[:, itau]) 
    #_coeff_phistar[:, itau] = avg_rho * dtau * _g * np.exp(2. * np.pi * 1j * itau / ntau)

# Precompute any additional force containers 
# Use the appropriate time stepper  
#if(not _ETD):
# For EM, need to add the linear part of the force to the nonlinear force container 
tmp = np.zeros((N_spatial, ntau), dtype=np.complex_)
tmpstar = np.zeros((N_spatial, ntau), dtype=np.complex_)
# Fill the CS-field containers 
for j in range(0, ntau):
  tmp[:, j] = A_nk(j, ntau, beta, _k2_grid, _lambda, ensemble, _mu) + _coeff_phi[:,j] 
  tmpstar[:, j] = np.conj(tmp[:, j]) + _coeff_phistar[:,j] 

fill_forces(phi, phistar, dSdphistar, dSdphi, ntau, _psi, ensemble, _g, beta, _coeff_phi, _coeff_phistar, _isShifting, tmp) 
# Only if _coeff_phi and _coeff_phistar are CL-time independent
for j in range(0, ntau):
  lincoef[:, j] = A_nk(j, ntau, beta, _k2_grid, _lambda, ensemble, _mu) + _coeff_phi[:, j]
  # shift any negative real and nonzero imag part away 
  lincoef_phistar[:, j] = np.conj(A_nk(j, ntau, beta, _k2_grid, _lambda, ensemble, _mu)) + _coeff_phistar[:, j]
  # Correct diverging terms by using Euler limit of ETD
  # Python's FFT accounts for scaling, i.e. ifft(fft(a) == a , therefore, take out the scaling factors  
  for m in range(0, N_spatial): 
    if(lincoef[m, j] == 0.):
      nonlincoef[m, j] = -1. * ntau * dt 
      nonlincoef_phistar[m, j] = -1. * ntau * dt 
      noisescl[m, j] = np.sqrt(ntau * dt)
      noisescl_phistar[m, j] = np.sqrt(ntau * dt)
      #noisescl[m, j] = np.sqrt(ntau * dt / dV)
      #noisescl_phistar[m, j] = np.sqrt(ntau * dt / dV)
    else: 
      nonlincoef[m, j] = (np.exp(-lincoef[m,j] * ntau * dt) - 1.)/lincoef[m, j]
      nonlincoef_phistar[m, j] = (np.exp(-lincoef_phistar[m,j] * ntau * dt) - 1.)/lincoef_phistar[m, j]
      noisescl[m, j] = np.sqrt((1. - np.exp(-2. * lincoef[m, j] * ntau * dt))/(2. * lincoef[m, j]))
      noisescl_phistar[m, j] = np.sqrt((1. - np.exp(-2. * lincoef_phistar[m, j] * ntau * dt))/(2. * lincoef_phistar[m, j]))
      #noisescl[m, j] = np.sqrt((1. - np.exp(-2. * lincoef[m, j] * ntau * dt))/(2. * lincoef[m, j] * dV))
      #noisescl_phistar[m, j] = np.sqrt((1. - np.exp(-2. * lincoef_phistar[m, j] * ntau * dt))/(2. * lincoef_phistar[m, j] * dV))
  lincoef[:, j] = np.exp(- lincoef[:, j] * ntau * dt)
  lincoef_phistar[:, j] = np.exp(- lincoef_phistar[:, j] * ntau * dt)


N_tot_avg = 0. + 1j*0 
N2_avg = 0. + 1j*0

e_residual_avg = 0 + 1j*0

ctr = 1

start = time.time()




# Timestep using ETD 
for l in range(0, numtsteps + 1):

  # Update the nonlinear forces before stepping; fill forces clears dSdphistar and dSdphi 
  fill_forces(phi, phistar, dSdphistar, dSdphi, ntau, _psi, ensemble, _g, beta, _coeff_phi, _coeff_phistar, _isShifting, tmp) 
  # Refill the linear coefficients, accounting for the shift via the nonlinear, linearized portion  
  for j in range(0, ntau):
    lincoef[:, j] = A_nk(j, ntau, beta, _k2_grid, _lambda, ensemble, _mu) + _coeff_phi[:, j]
    # shift any negative real and nonzero imag part away 
    lincoef_phistar[:, j] = np.conj(A_nk(j, ntau, beta, _k2_grid, _lambda, ensemble, _mu)) + _coeff_phistar[:, j]
    # Correct diverging terms by using Euler limit of ETD
    # Python's FFT accounts for scaling, i.e. ifft(fft(a) == a , therefore, take out the scaling factors  
    for m in range(0, N_spatial): 
      if(lincoef[m, j] == 0.):
        nonlincoef[m, j] = -1. * ntau * dt 
        nonlincoef_phistar[m, j] = -1. * ntau * dt 
        noisescl[m, j] = np.sqrt(ntau * dt)
        noisescl_phistar[m, j] = np.sqrt(ntau * dt)
        #noisescl[m, j] = np.sqrt(ntau * dt / dV)
        #noisescl_phistar[m, j] = np.sqrt(ntau * dt / dV)
      else: 
        nonlincoef[m, j] = (np.exp(-lincoef[m,j] * ntau * dt) - 1.)/lincoef[m, j]
        nonlincoef_phistar[m, j] = (np.exp(-lincoef_phistar[m,j] * ntau * dt) - 1.)/lincoef_phistar[m, j]
        noisescl[m, j] = np.sqrt((1. - np.exp(-2. * lincoef[m, j] * ntau * dt))/(2. * lincoef[m, j]))
        noisescl_phistar[m, j] = np.sqrt((1. - np.exp(-2. * lincoef_phistar[m, j] * ntau * dt))/(2. * lincoef_phistar[m, j]))
        #noisescl[m, j] = np.sqrt((1. - np.exp(-2. * lincoef[m, j] * ntau * dt))/(2. * lincoef[m, j] * dV))
        #noisescl_phistar[m, j] = np.sqrt((1. - np.exp(-2. * lincoef_phistar[m, j] * ntau * dt))/(2. * lincoef_phistar[m, j] * dV))
    lincoef[:, j] = np.exp(- lincoef[:, j] * ntau * dt)
    lincoef_phistar[:, j] = np.exp(- lincoef_phistar[:, j] * ntau * dt)

  # For ETD, can proceed; for EM, must add linear force contributions to the conatiners  
  if(not _ETD):
    # d+1 FFT force container
    dSdphistar = fft_dp1(dSdphistar) 
    dSdphi = fft_dp1(dSdphi) 
    # Add linearized contributions
    dSdphistar += tmp * fft_dp1(phi)
    #dSdphi += np.conj(tmp) * fft_dp1(phistar)
    dSdphi += tmpstar * fft_dp1(phistar)

    # need to iFFT phi and phistar
    phi = ifft_dp1(phi) 
    phistar = ifft_dp1(phistar)

    # inverse d+1 FFT force container
    dSdphistar = ifft_dp1(dSdphistar) 
    dSdphi = ifft_dp1(dSdphi) 

    # Do EM step 
    if(_do_implicit):
      phi, phistar = EM_implicit(phi, phistar, dSdphistar, dSdphi, _isOffDiagonal, _CLnoise, 1., dt, tmp, tmpstar)
    else:
      phi, phistar = EM(phi, phistar, dSdphistar, dSdphi, _isOffDiagonal, _CLnoise, 1., dt)
  else:  
    if(_do_implicit): 
      phi, phistar = ETD_implicit(phi, phistar, dSdphistar, dSdphi, lincoef, lincoef_phistar, nonlincoef, nonlincoef_phistar, noisescl, noisescl_phistar, _CLnoise)
    else:
      phi, phistar = ETD(phi, phistar, dSdphistar, dSdphi, lincoef, lincoef_phistar, nonlincoef, nonlincoef_phistar, noisescl, noisescl_phistar, _CLnoise)


  # ---- Calculate and Update Observables ------- 

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

  e_residual_avg = constraint_err(N_input, phi, phistar)/iofreq

  t += dt

  # Output on interval
  if(l % iofreq == 0 and l > 0):
     if(ctr %  25):
       print("Completed {} of {} steps".format(l, numtsteps))
     # opout.write("{} {} {} {} {}\n".format(it, Msum.real / Navg, Msum.imag / Navg, psi.real, psi.imag))
     t_s[ctr] = t 
     N_tot_s[ctr] = N_tot_avg 
     N2_s[ctr] = N2_avg

     if(ensemble == "CANONICAL"):
       psi_s[ctr] = _psi
       print('Constraint Residual: ' + str(e_residual_avg))
     # clear the averages 
     N2_avg = 0. + 1j*0 
     N_tot_avg = 0. + 1j*0 
     e_residual_avg = 0. + 1j*0 
     ctr += 1

    

end = time.time()
print()
print()
if(l == numtsteps):
  print('Simulation finished: Runtime = ' + str(end - start) + ' seconds')




# Print the results (noise long-time averages)
print()
print()
print('The Boson Particle Number is: ' + str(np.mean(N_tot_s[4:ctr].real)))
print()
print('The Particle Number squared is: ' + str(np.mean(N2_s[4:ctr].real)))
print()
print('The density is: ' + str(np.mean(N_tot_s[10:ctr].real)/Vol))
print()

# plot the results 

plt.figure(1)
plt.title('Particle Number: CL Simulation', fontsize = 20, fontweight = 'bold')
plt.plot(t_s[0:ctr], N_tot_s[0:ctr].real, '*-', color = 'green', linewidth = 0.5, label = 'Samples: real')
plt.plot(t_s[0:ctr], N_tot_s[0:ctr].imag, '*-', color = 'skyblue', linewidth=0.5,label = 'Samples: imag')
#plt.plot(t_s, np.ones(len(t_s)), 'k', label = 'Constraint')
plt.xlabel('CL time', fontsize = 20, fontweight = 'bold')
plt.ylabel('$N_{tot}$', fontsize = 20, fontweight = 'bold') 
#plt.ylim([-5, 10])
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
plt.plot(t_s[0:ctr], N2_s[0:ctr].real, '*-', color = 'purple', label = 'Samples')
plt.plot(t_s[0:ctr], N2_s[0:ctr].imag, '*-', color = 'skyblue', label = 'Samples')
# plt.plot(t_s, np.ones(len(t_s)), 'k', label = 'Constraint')
plt.xlabel('CL time', fontsize = 20, fontweight = 'bold')
plt.ylabel('$N^2$', fontsize = 20, fontweight = 'bold') 
#plt.ylim([-5, 20])
plt.legend()
plt.show()

