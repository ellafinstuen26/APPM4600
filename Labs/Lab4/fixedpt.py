# import libraries
import numpy as np
    
def driver():

    f0 = lambda x: x-(x**5-7)/(5*x**4)
    f1 = lambda x: (10/(x+4))**(1/2)

    Nmax = 100
    tol = 1e-10

    #prelab test
    x0 = 1
    [x,xstar,ier] = fixed_pt(f0,x0,tol,Nmax)
    print('the approximate fixed point is:',xstar)
    print('approximations: ', x,)
    print('f1(xstar):',f0(xstar))
    print('Error message reads:',ier)
    [fit,diff1,diff2] = convergence_order(x,xstar)

    #Exercise 2
    x0 = 1.5
    [x,xstar,ier] = fixed_pt(f1,x0,tol,Nmax)
    print('the approximate fixed point is:',xstar)
    print('approximations: ', x, len(x))
    print('f1(xstar):',f1(xstar))
    print('Error message reads:',ier)
    [fit,diff1,diff2] = convergence_order(x,xstar)

    #Exercise 3
    p = aitkens(x,tol,Nmax)
    [fit,diff1,diff2] = convergence_order(p,xstar)





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
    
def fixed_pt(f,x0,tol,Nmax):

    ''' x0 = initial guess''' 
    ''' Nmax = max number of iterations'''
    ''' tol = stopping tolerance'''
    x = np.zeros((Nmax+1,1))
    x[0] = x0

    count = 0
    while (count <Nmax):
       count = count +1
       x1 = f(x0)
       x[count] = x1
       if (abs(x1-x0) <tol):
          xstar = x1
          ier = 0
          return [x[:count],xstar,ier]
       x0 = x1

    xstar = x1
    ier = 1
    return [x[:count],xstar, ier]

def convergence_order(x,xstar):
    diff1 = np.abs(x[1::]-xstar)
    diff2 = np.abs(x[0:-1]-xstar)
    fit = np.polyfit(np.log(diff2.flatten()),np.log(diff1.flatten()),1)
    print('the order equation is')
    print('log(|p_{n+1}-p|) = log(lambda) + alpha*log(|p_n-p|) where')
    print('lambda = ', str(np.exp(fit[1])))
    print('alpha = ', str(fit[0]))

    return [fit,diff1,diff2]


# p = [p_{n+2}*p_{n}-p_{n+1}^2]/[-2p_{n+1} + p_{n+2} + p_{n})]
def aitkens(x,tol,Nmax):
    x0 = x[:-2]
    x1 = x[1:-1]
    x2 = x[2:]
    diff1 = x2*x0-x1**2
    diff2 = -2*x1+x2+x0

    return diff1/diff2

driver()