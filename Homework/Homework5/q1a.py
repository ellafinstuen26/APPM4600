import math
import scipy
from scipy import special
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv 
from numpy.linalg import norm 

def driver():
    f = lambda x,y: int(3*x**2-y**2)
    g = lambda x,y: int(3*x*y**2-x**3-1)
    tol = 1e-13
    Nmax = 100
    m = np.matrix([[1/6, 1/18], [0, 1/6]])
    x0 = np.array([1, 1])

    [xstar,ier,its] = system(x0,m,tol,Nmax)
    x = xstar[0]
    y = xstar[1]

    print('The solution is:')
    print('x =' ,x, '\ny =',y)
    print('f(x,y) = ', f(x,y))
    print('g(x,y) = ', g(x,y))
    print('Number of iterations is:',its)
    print('The error message reads:',ier) 


def evalF(p): 

    F = np.zeros(2)
    x = p[0]
    y = p[1]
    F[0] = 3*x**2-y**2
    F[1] = 3*x*y**2-x**3-1
    
    return F


def system(x0,m,tol,Nmax):

    ''' Lazy Newton = use only the inverse of the Jacobian for initial guess'''
    ''' inputs: x0 = initial guess, tol = tolerance, Nmax = max its'''
    ''' Outputs: xstar= approx root, ier = error message, its = num its'''

    for its in range(Nmax):

       F = evalF(x0)
       mdotF = m.dot(F).reshape(2,1)
       mF = np.array([float(mdotF[0]),float(mdotF[1])])

       x1 = x0 - mF
       
       if (norm(x1-x0) < tol):
           xstar = x1
           ier =0
           return[xstar, ier,its]
       x0 = x1
    
    xstar = x1
    ier = 1
    return[xstar,ier,its]  

        
driver()