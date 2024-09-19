# import libraries
import numpy as np
    
def driver():

# test functions 
#      f1 = lambda x: 1+0.5*np.sin(x)
# # fixed point is alpha1 = 1.4987....

#      f2 = lambda x: 3+2*np.sin(x)
#fixed point is alpha2 = 3.09... 

     f1 = lambda x: -np.sin(2*x)+5*x/4-3/4

     Nmax = 1000
     tol = 1e-11

#following tests are to try to approximate the other roots

#test 1 '''
     x0 = 4.6
     [xstar,ier] = fixedpt(f1,x0,tol,Nmax)
     print('the approximate first fixed point is:',xstar)
     print('f1(xstar):',f1(xstar))
     print('Error message reads:',ier)

#test 1 '''
     x0 = 4
     [xstar,ier] = fixedpt(f1,x0,tol,Nmax)
     print('the approximate second fixed point is:',xstar)
     print('f1(xstar):',f1(xstar))
     print('Error message reads:',ier)

#test 2 '''
     x0 = 1.7
     [xstar,ier] = fixedpt(f1,x0,tol,Nmax)
     print('the approximate third fixed point is:',xstar)
     print('f1(xstar):',f1(xstar))
     print('Error message reads:',ier)

#test 2 '''
     x0 = -1.4
     [xstar,ier] = fixedpt(f1,x0,tol,Nmax)
     print('the approximate fourth fixed point is:',xstar)
     print('f1(xstar):',f1(xstar))
     print('Error message reads:',ier)
    



# define routines
def fixedpt(f,x0,tol,Nmax):

    ''' x0 = initial guess''' 
    ''' Nmax = max number of iterations'''
    ''' tol = stopping tolerance'''

    count = 0
    while (count <Nmax):
       count = count +1
       x1 = f(x0)
       if (abs(x1-x0) <tol):
          xstar = x1
          ier = 0
          return [xstar,ier]
       x0 = x1

    xstar = x1
    ier = 1
    return [xstar, ier]
    

driver()