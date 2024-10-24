import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

def driver():


    f = lambda x: 1./(1.+x**2)
    a = -5
    b = 5
    NS = [5,10,15,20]
    Neval = 1000
    xeval = np.linspace(a,b,Neval+1)
    yeval_l = []
    fex = f(xeval)

    for n in range(4):
       xint = np.linspace(a,b,NS[n]+1)

       yint = f(xint)

       yeval = np.zeros(Neval+1)

       for kk in range(Neval+1):
          yeval[kk] = eval_lagrange(xeval[kk],xint,yint,NS[n])

       yeval_l.append(yeval)
       

    plt.figure()    
    plt.plot(xeval,fex,'rs-', label='Exact function')
    plt.plot(xeval,yeval_l[0],'o--', label='N = 5') 
    plt.plot(xeval,yeval_l[1],'o--', label='N = 10') 
    plt.plot(xeval,yeval_l[2],'o--', label='N = 15') 
    plt.plot(xeval,yeval_l[3],'o--', label='N = 20') 
    plt.title("Lagrange Interpolation for various N and Exact function")
    plt.legend()

    plt.figure() 
    err_l0 = abs(yeval_l[0]-fex)
    err_l1 = abs(yeval_l[1]-fex)
    err_l2 = abs(yeval_l[2]-fex)
    err_l3 = abs(yeval_l[3]-fex)
    plt.semilogy(xeval,err_l0,'o--', label='N = 5')
    plt.semilogy(xeval,err_l1,'o--', label='N = 10')
    plt.semilogy(xeval,err_l2,'o--', label='N = 15')
    plt.semilogy(xeval,err_l3,'o--', label='N = 20')
    plt.legend()

    plt.title("Absolute Error of Lagrange Interpolation")
    plt.show()

def eval_lagrange(xeval,xint,yint,N):

    lj = np.ones(N+1)
    
    for count in range(N+1):
       for jj in range(N+1):
           if (jj != count):
              lj[count] = lj[count]*(xeval - xint[jj])/(xint[count]-xint[jj])

    yeval = 0.
    
    for jj in range(N+1):
       yeval = yeval + yint[jj]*lj[jj]
  
    return(yeval)

       

driver()        
