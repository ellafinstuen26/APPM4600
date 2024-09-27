# import libraries
import math
import scipy
from scipy import special
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
from scipy.stats import linregress
        
def driver():
    f = lambda x: x**6-x-1
    fp = lambda x: 6*x**5-1
    p0 = 2
    p1 = 1

    Nmax = 100
    tol = 1.e-13

    (p_n,e_n,pstar1,info1,it1) = newton(f,fp,p0,tol, Nmax)
    print("Approximation of root with Newton's Method")
    print('the approximate root is', '%16.16e' % pstar1)
    print('the error message reads:', '%d' % info1)
    print('Number of iterations:', '%d' % it1)

    (p_s,e_s,pstar2,info2,it2) = secant(f,p0,p1,tol, Nmax)
    print("Approximation of root with Secant Method")
    print('the approximate root is', '%16.16e' % pstar2)
    print('the error message reads:', '%d' % info2)
    print('Number of iterations:', '%d' % it2)

    m = max(it1,it2)+1
    errors = [[e_n[i],e_s[i]] for i in range(m+1)]
    for error in errors[m-1:]:
        if error[0] == 0:
          error[0] = " "
        if error[1] == 0:
          error[1] = " "

    print(tabulate(errors, headers=['Error for Newton', 'Error for Secant']))
    
    e_slope_n = e_n[:it1+1]
    e_slope_s = e_s[:it2]
    plt.loglog(e_slope_n[:-1], e_slope_n[1:],label = "Newton's Method", color = 'blue')
    plt.loglog(e_slope_s[:-1], e_slope_s[1:],label = "Secant Method", color = 'pink')
    plt.ylabel("log|x_{k}-x*|")
    plt.xlabel("log|x_{k+1}-x*|")
    plt.title('Convergence of Newton and Secant Methods (log-log scale)')
    plt.legend()
    plt.show()

    # Perform linear regression to calculate the slope for slopes>1. Switch indices for slopes<1
    slope_newton, intercept, r_value, p_value, std_err = linregress(np.log(e_slope_n[:-1]), np.log(e_slope_n[1:]))
    slope_secant, intercept, r_value, p_value, std_err = linregress(np.log(e_slope_s[:-1]), np.log(e_slope_s[1:]))

    # Print the slopes (order of convergence)
    print(f"Slope (Order of Convergence) for Newton's Method: {slope_newton}")
    print(f"Slope (Order of Convergence) for Secant Method: {slope_secant}")




def newton(f,fp,p0,tol,Nmax):
  """
  Newton iteration.
  
  Inputs:
    f,fp - function and derivative
    p0   - initial guess for root
    tol  - iteration stops when p_n,p_{n+1} are within tol
    Nmax - max number of iterations
  Returns:
    p     - an array of the iterates
    pstar - the last iterate
    info  - success message
          - 0 if we met tol
          - 1 if we hit Nmax iterations (fail)
     
  """
  p = np.zeros(Nmax+1);
  e = np.zeros(Nmax+1);
  p[0] = p0
  e[0] = p0
  for it in range(Nmax):
      p1 = p0-f(p0)/fp(p0)
      p[it+1] = p1
      e[it+1] = abs(p1-p0)
      if (abs(p1-p0) < tol):
          pstar = p1
          info = 0
          return [p,e,pstar,info,it]
      p0 = p1
  pstar = p1
  info = 1
  return [p,e,pstar,info,it]

def secant(f,p0,p1,tol,Nmax):
  """
  Secant iteration.
  
  Inputs:
    f - function
    p0,p1   - two initial guess for root
    tol  - iteration stops when p_n,p_{n+1} are within tol
    Nmax - max number of iterations
  Returns:
    p     - an array of the iterates
    pstar - the last iterate
    ier  - success message
          - 0 if we met tol
          - 1 if we hit Nmax iterations (fail)
     
  """
  p = np.zeros(Nmax+1);
  e = np.zeros(Nmax+1)
  p[0] = p0
  p[1] = p1
  e[0] = abs(p1-p0)
  if abs(f(p0)-f(p1))<tol:
    ier = 1
    pstar = p1
    return [p,e,pstar,ier,0]
  
  for it in range(1,Nmax+1):
    p2 = p1 - (f(p1)*(p1-p0))/(f(p1)-f(p0))
    p[it+1] = p2
    if it != 1:
        e[it-1] = abs(p2-p1)
    if abs(p2-p1)<tol:
        pstar = p2
        ier = 0
        return [p,e,pstar,ier,it]
    p0 = p1
    p1 = p2
    
  pstar = p2
  ier = 1
  return [p,e,pstar,ier,it]
        
driver()
