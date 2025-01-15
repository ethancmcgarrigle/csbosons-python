import numpy as np
from scipy.optimize import root
from typing import Callable, Tuple, Union
import warnings

class NonlinearPeriodicSolver:
    def __init__(self, N: int):
        """
        Initialize solver for N coupled nonlinear equations with periodic boundary conditions.
        
        Args:
            N: Number of equations/variables
        """
        self.N = N
    
    def _wrap_index(self, j: int) -> int:
        """Handle periodic boundary conditions by wrapping indices."""
        return j % self.N
    
    def construct_residuals(self, 
                          x: np.ndarray, 
                          C: np.ndarray, 
                          B_plus: np.ndarray, 
                          B_minus: np.ndarray, 
                          A: np.ndarray) -> np.ndarray:
        """
        Construct the residual equations g_j for the system.
        
        Args:
            x: Complex array of variables [x_0, ..., x_{N-1}]
            C: Complex array of C coefficients
            B_plus: Complex array of B^+ coefficients
            B_minus: Complex array of B^- coefficients
            A: Complex array of A coefficients
            
        Returns:
            Array of residuals g_j
        """
        g = np.zeros(self.N, dtype=complex)
        
        for j in range(self.N):
            j_plus = self._wrap_index(j + 1)
            j_minus = self._wrap_index(j - 1)
            
            g[j] = (C[j] + B_plus[j] * x[j_plus] + B_minus[j] * x[j_minus] + A[j] * x[j_plus] * x[j_minus])
            
        return g
    
    def _split_complex(self, z: np.ndarray) -> np.ndarray:
        """Split complex array into real array with real and imaginary parts."""
        return np.concatenate([z.real, z.imag])
    
    def _join_complex(self, z: np.ndarray) -> np.ndarray:
        """Join real and imaginary parts back into complex array."""
        n = len(z) // 2
        return z[:n] + 1j * z[n:]
    
    def _residual_function(self, x_real: np.ndarray, 
                          C: np.ndarray, 
                          B_plus: np.ndarray, 
                          B_minus: np.ndarray, 
                          A: np.ndarray) -> np.ndarray:
        """Wrapper function for scipy.optimize.root that handles complex numbers."""
        x_complex = self._join_complex(x_real)
        residuals_complex = self.construct_residuals(x_complex, C, B_plus, B_minus, A)
        return self._split_complex(residuals_complex)
    
    def solve(self, 
             C: np.ndarray, 
             B_plus: np.ndarray, 
             B_minus: np.ndarray, 
             A: np.ndarray, 
             x0: np.ndarray = None, 
             method: str = 'hybr',
             **kwargs) -> Tuple[np.ndarray, bool]:
        """
        Solve the system of nonlinear equations.
        
        Args:
            C, B_plus, B_minus, A: Arrays of coefficients
            x0: Initial guess (if None, random initialization is used)
            method: Solver method for scipy.optimize.root
            **kwargs: Additional arguments passed to scipy.optimize.root
            
        Returns:
            Tuple of (solution array, success boolean)
        """
        if x0 is None:
            x0 = np.random.rand(self.N) + 1j * np.random.rand(self.N)
            
        x0_real = self._split_complex(x0)
        
        result = root(self._residual_function, x0_real, 
                     args=(C, B_plus, B_minus, A),
                     method=method, **kwargs)
        
        if not result.success:
            warnings.warn(f"Solver failed to converge: {result.message}")
            
        solution = self._join_complex(result.x)
        return solution, result.success

 #    def solve(self, 
 #             C: np.ndarray, 
 #             B_plus: np.ndarray, 
 #             B_minus: np.ndarray, 
 #             A: np.ndarray, 
 #             x0: np.ndarray = None,
 #             method: str = 'hybr',
 #             **kwargs) -> Tuple[np.ndarray, bool]:
 #        """
 #        Solve the system of nonlinear equations directly with complex numbers.
 #        """
 #        if x0 is None:
 #            x0 = np.random.rand(self.N) + 1j * np.random.rand(self.N)
 #            
 #        result = root(self.construct_residuals, x0, 
 #                     args=(C, B_plus, B_minus, A),
 #                     method=method, **kwargs)
 #        
 #        return result.x, result.success

    
    def verify_solution(self, 
                       x: np.ndarray, 
                       C: np.ndarray, 
                       B_plus: np.ndarray, 
                       B_minus: np.ndarray, 
                       A: np.ndarray, 
                       tol: float = 1e-10) -> bool:
        """
        Verify if a solution satisfies the equations within tolerance.
        
        Args:
            x: Solution to verify
            C, B_plus, B_minus, A: Coefficient arrays
            tol: Tolerance for residuals
            
        Returns:
            Boolean indicating if solution is valid
        """
        residuals = self.construct_residuals(x, C, B_plus, B_minus, A)
        max_residual = np.max(np.abs(residuals))
        return max_residual < tol
