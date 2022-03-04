import numpy as np
from numpy import linalg
import yaml
import math
import matplotlib.pyplot as plt 
import time
from scipy.fft import fft 
from scipy.fft import ifft
import matplotlib 
import matplotlib.pyplot as plt 
import copy
import io_functions
import sys


def grad_descent(A, b, x0, stepsize, max_iterations, ETOL, print_iters):
  # Gradient descent method that uses initial guess x_0 in solving Ax = b linear problem:
  # Takes in the maximum number of iterations and the error tolerance as inputs  
  x_i = np.zeros(len(x0), dtype = np.complex_)
  # initial iteration, set x_i = x_0
  x_i += x0

  for i in range(0, max_iterations + 1):
     # 1. Calculate the gradient (Ax - b) 
     gradient = A @ x_i - b
     # 2. Check for convergence via the 2-norm
     if(np.linalg.norm(gradient, 2) < ETOL):
       break     

     x_i -= stepsize * gradient 
     # sol = x_i

  if(print_iters):
    print('Number of iteratons for 1 gradient descent loop: ')
    print(i + 1)
    print()

  return x_i 




