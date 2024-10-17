import matplotlib.pyplot as plt
import numpy as np
import math
from numpy.linalg import inv
def driver():
    f = lambda x: 1/(1+(10*x)**2)
    a = -1
    b = 1
    # create points you want to evaluate at
    Neval = 100
    xeval = np.linspace(a,b,Neval)
    # number of intervals’’’
    Nint = 10
    # evaluate the linear spline’’’
    yeval = eval_lin_spline(xeval,Neval,a,b,f,Nint)

    # evaluate f at the evaluation points’’’
    fex = f(xeval)
    plt.figure()
    plt.plot(xeval,fex,'r-',linewidth=4)
    plt.plot(xeval,yeval,'bo-')
    plt.legend()
    plt.show
    err = abs(yeval-fex)
    plt.figure()
    plt.plot(xeval,err,'r-')
    plt.show()


def eval_lin_spline(xeval,Neval,a,b,f,Nint):
    # create the intervals for piecewise approximations’’’
    xint = np.linspace(a,b,Nint+1)
    # create vector to store the evaluation of the linear splines’’’
    yeval = np.zeros(Neval)
    for j in range(Nint):
    #find indices of xeval in interval (xint(jint),xint(jint+1))’’’
    #let ind denote the indices in the intervals’’’
        atmp = xint[j]
        btmp= xint[j+1]
        # find indices of values of xeval in the interval
        ind= np.where((xeval >= atmp) & (xeval <= btmp))
        xloc = xeval[ind]
        n = len(xloc)
        #temporarily store your info for creating a line in the interval of interest
        fa = f(atmp)
        fb = f(btmp)
        yloc = np.zeros(len(xloc))
        for kk in range(n):
        #use your line evaluator to evaluate the spline at each location
            a = xloc[kk]
            yloc[kk] = evaluate(atmp,btmp,f,a)
            # Copy yloc into the final vector
            yeval[ind] = yloc

    return yeval

def evaluate(x0,x1,f,a):
    y0 = f(x0)
    y1 = f(x1)
  
    m = (y1 - y0) / (x1 - x0)
  
    
    return y0 + m * (a - x0)

driver()