import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

def driver():


    f = lambda x: 1/(1+(10*x)**2)

    N = 11
    ''' interval'''
    a = -1
    b = 1
   
   
    ''' create equispaced interpolation nodes'''
    xint = np.array([np.cos((2*i-1)*np.pi/(2*N)) for i in range(1,N+1)])
    
    ''' create interpolation data'''
    yint = f(xint)

    ''' create weights '''
    w = weights(xint)
    
    ''' create points for evaluating '''
    Neval = 1001
    xeval = np.linspace(a,b,Neval+1)
    yeval_bary = barycentric_lagrange(xeval, xint, yint, w)
          


    ''' create vector with exact values'''
    fex = f(xeval)
       

    plt.figure()    
    plt.plot(xeval,fex,'ro-',label='Exact Function')
    plt.plot(xeval,yeval_bary,'bo--',label='Barycentric') 
    plt.legend()
    plt.legend()

    plt.figure() 
    err_l = abs(yeval_bary-fex)
    plt.semilogy(xeval,err_l,'ro--')
    plt.title("Error from Barycentric vs exact function")
    plt.show()

def weights(xint):

    n = len(xint)
    w = np.ones(n)
    
    for j in range(n):
        for i in range(n):
            if i != j:
                w[j] *= 1 / (xint[j] - xint[i])
    
    return w

def barycentric_lagrange(xeval, xint, yint, w):
    n = len(xint)
    yeval = np.zeros_like(xeval)
    
    for k in range(len(xeval)):
        x = xeval[k]
        
        if x in xint:
            yeval[k] = yint[np.where(xint == x)[0][0]]
        else:
            numerator = np.sum(w / (x - xint) * yint)
            denominator = np.sum(w / (x - xint))
            yeval[k] = numerator / denominator
    
    return yeval

driver()