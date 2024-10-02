import math
import scipy
from scipy import special
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv 
from numpy.linalg import norm 

def driver():

    tol = 1e-13
    Nmax = 100
    x0 = np.array([1, 1, 1])

    [xs,xstar,ier,its] = system(x0,tol,Nmax)
    x = xstar[0]
    y = xstar[1]
    z = xstar[2]

    print('The solution is:')
    print('x =' ,x, '\ny =',y, '\nz =',z)
    print('Number of iterations is:',its)
    print('The error message reads:',ier) 

    convergence_order(xs,xstar)



def evalIt(p): 

    F = np.zeros(3)
    x = p[0]
    y = p[1]
    z = p[2]
    F[0] = (x**2+4*y**2+4*z**2-16)*(2*x)/((2*x)**2+(8*y)**2+(8*z)**2)

    F[1] = (x**2+4*y**2+4*z**2-16)*(8*y)/((2*x)**2+(8*y)**2+(8*z)**2)

    F[2] = (x**2+4*y**2+4*z**2-16)*(8*z)/((2*x)**2+(8*y)**2+(8*z)**2)
    
    return F

def system(x0,tol,Nmax):

    ''' inputs: x0 = initial guess, tol = tolerance, Nmax = max its'''
    ''' Outputs: xstar= approx root, ier = error message, its = num its'''
    xs = [x0]

    for its in range(Nmax):
       
       F = evalIt(x0)
       
       x1 = x0 - F
       xs.append(x1)
       
       if (norm(x1-x0) < tol):
           xstar = x1
           ier = 0
           return[xs,xstar, ier, its]
           
       x0 = x1
    
    xstar = x1
    ier = 1
    return[xs,xstar,ier,its]

def convergence_order(x,xstar):
    diff1 = norm(x[1::]-xstar)
    diff2 = norm(x[0:-1]-xstar)
    fit = np.polyfit(np.log(diff2.flatten()),np.log(diff1.flatten()),1)
    print('the order equation is')
    print('log(|p_{n+1}-p|) = log(lambda) + alpha*log(|p_n-p|) where')
    print('lambda = ', str(np.exp(fit[1])))
    print('alpha = ', str(fit[0]))

    return [fit,diff1,diff2]



# def system(x0,m,tol,Nmax):

#     for its in range(Nmax):

#        F = evalF(x0)
#        mdotF = m.dot(F).reshape(2,1)
#        mF = np.array([float(mdotF[0]),float(mdotF[1])])

#        x1 = x0 - mF
       
#        if (norm(x1-x0) < tol):
#            xstar = x1
#            ier =0
#            return[xstar, ier,its]
#        x0 = x1
    
#     xstar = x1
#     ier = 1
#     return[xstar,ier,its]  

        
driver()