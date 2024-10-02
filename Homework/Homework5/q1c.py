import numpy as np
import math
import time
from numpy.linalg import inv 
from numpy.linalg import norm 

def driver():

    x0 = np.array([1,1])
    
    Nmax = 100
    tol = 1e-15
    
    t = time.time()
    for j in range(50):
      [xstar,ier,its] =  Newton(x0,tol,Nmax)
    elapsed = time.time()-t

    x = xstar[0]
    y = xstar[1]
    # f = evalF(xstar)
    print('The solution is:')
    print('x =' ,x, '\ny =',y)
    # print('f(x,y) = ', f[0])
    # print('g(x,y) = ', f[1])
    print('Newton: the error message reads:',ier) 
    print('Newton: took this many seconds:',elapsed/50)
    print('Netwon: number of iterations is:',its)
     
     
def evalF(p): 

    F = np.zeros(2)

    x = p[0]
    y = p[1]

    F[0] = 3*x**2-y**2
    F[1] = 3*x*y**2-x**3-1
    return F
    
def evalJ(p): 
    x = p[0]
    y = p[1]
    
    J = np.array([[6*x, -2*y], 
        [3*y**2-3*x**2, 6*x*y]])
    
    return J


def Newton(x0,tol,Nmax):

    ''' inputs: x0 = initial guess, tol = tolerance, Nmax = max its'''
    ''' Outputs: xstar= approx root, ier = error message, its = num its'''

    for its in range(Nmax):
       J = evalJ(x0)
       Jinv = inv(J)
       F = evalF(x0)
       
       x1 = x0 - Jinv.dot(F)
       
       if (norm(x1-x0) < tol):
           xstar = x1
           ier =0
           return[xstar, ier, its]
           
       x0 = x1
    
    xstar = x1
    ier = 1
    return[xstar,ier,its]
           
driver()