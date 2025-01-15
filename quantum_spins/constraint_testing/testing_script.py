import numpy as np
import matplotlib.pyplot as plt
from nonlinear_solver import NonlinearPeriodicSolver  # Assuming previous code is saved as nonlinear_solver.py

def import_CSfield_1site(filename):
  ''' Imports a CSfield with 1 spatial degree of freedom and ntau "imaginary time" points'''
  ''' Returns a complex-valued vector of length ntau ''' 
  raw_data = np.loadtxt(filename, unpack=True)
  
  raw_data = raw_data[1:] # exclude first entry showing spatial location 
  ntau = int(len(raw_data)/2.)
  
  CSfield_1site_vector = np.zeros(ntau, dtype=np.complex128)
  for n in range(ntau):
    # In raw_data, real part stored in first entry, imaginary part in 2nd, repeats until 2ntau 
    CSfield_1site_vector[n] = raw_data[2*n] + 1j*raw_data[2*n+1]
  
  return CSfield_1site_vector 


# Set random seed for reproducibility
#np.random.seed(42)

# Create a specific set of coefficients
# Using a physical-inspired example where coefficients have some structure
# rather than being completely random

# Base coefficients with some structure
C = import_CSfield_1site('C_coeff.dat') 
B_plus = import_CSfield_1site('B+_coeff.dat') 
B_minus = import_CSfield_1site('B-_coeff.dat') 
A = import_CSfield_1site('A_coeff.dat')

N = len(A)
# Initialize system size and solver
solver = NonlinearPeriodicSolver(N)

# Try different initial conditions
x0_attempts = [
    np.ones(N),  # Uniform initial guess of one
    np.zeros(N),  # Uniform initial guess of zero 
    #np.exp(1j * 2 * np.pi * np.arange(N) / N),  # Phase-winding initial guess
    np.random.rand(N) + 1j * np.random.rand(N)  # Random initial guess
]

results = []
for i, x0 in enumerate(x0_attempts):
    solution, success = solver.solve(C, B_plus, B_minus, A, x0=x0, method='lm')
    is_valid = solver.verify_solution(solution, C, B_plus, B_minus, A)
    residuals = solver.construct_residuals(solution, C, B_plus, B_minus, A)
    
    results.append({
        'initial_guess': x0,
        'solution': solution,
        'success': success,
        'is_valid': is_valid,
        'max_residual': np.max(np.abs(residuals)),
        'attempt': i + 1
    })

# Print results and analysis
for res in results:
    print(f"\nAttempt {res['attempt']}:")
    print(f"Solver success: {res['success']}")
    print(f"Solution valid: {res['is_valid']}")
    print(f"Max residual: {res['max_residual']:.2e}")
    print(f"Solution magnitude range: [{np.min(np.abs(res['solution'])):.2f}, "
          f"{np.max(np.abs(res['solution'])):.2f}]")
    
# Plot the best solution
best_result = min(results, key=lambda x: x['max_residual'])
best_solution = best_result['solution']


print(best_solution)

plt.figure(figsize=(12, 4))

# Plot magnitude
plt.subplot(121)
plt.plot(best_solution.real, 'o-', label='Real part')
plt.xlabel('Site index j')
#plt.ylabel('|x_j|')
plt.ylabel('Re[x_j]')
#plt.title('Solution Magnitude')
plt.grid(True)

# Plot phase
plt.subplot(122)
plt.plot(best_solution.imag, 'o-', label='Imaginary part')
plt.xlabel('Site index j')
plt.ylabel('Im[x_j]')
#plt.title('Solution Phase')
plt.grid(True)

plt.tight_layout()
plt.show()

# Verify periodicity
 #periodicity_error = np.abs(best_solution[0] - best_solution[-1])
 #print(f"\nPeriodicity check - |x_0 - x_N|: {periodicity_error:.2e}")

# Print coefficient properties for reference
 #print("\nCoefficient properties:")
 #print(f"Max |C|: {np.max(np.abs(C)):.2f}")
 #print(f"Max |B_plus|: {np.max(np.abs(B_plus)):.2f}")
 #print(f"Max |B_minus|: {np.max(np.abs(B_minus)):.2f}")
 #print(f"Max |A|: {np.max(np.abs(A)):.2f}")
 #
