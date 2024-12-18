import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la
import math

def driver():

    f = lambda x: 1./(1.+x**2)
    fp = lambda x: -2*x/(1.+x**2)**2

    a = -5
    b = 5
    NS = [5,10,15,20]
    Neval = 1000
    xeval = np.linspace(a,b,Neval+1)
    yeval_H = []
    fex = f(xeval)

    for n in range(4):
       xint = np.array([5*np.cos((2*i-1)*np.pi/(2*NS[n])) for i in range(1,NS[n]+1)])

       yint = f(xint)
       ypint = fp(xint)

       yeval = np.zeros(Neval+1)

       for kk in range(Neval+1):
          yeval[kk] = eval_hermite(xeval[kk],xint,yint,ypint,NS[n]-1)

       yeval_H.append(yeval)
       

    plt.figure()    
    plt.plot(xeval,fex,'rs-', label='Exact function')
    plt.plot(xeval,yeval_H[0],'o--', label='N = 5') 
    plt.plot(xeval,yeval_H[1],'o--', label='N = 10') 
    plt.plot(xeval,yeval_H[2],'o--', label='N = 15') 
    plt.plot(xeval,yeval_H[3],'o--', label='N = 20') 
    plt.title("Hermite Interpolation for various N and Exact function")
    plt.legend()

    plt.figure() 
    err_l0 = abs(yeval_H[0]-fex)
    err_l1 = abs(yeval_H[1]-fex)
    err_l2 = abs(yeval_H[2]-fex)
    err_l3 = abs(yeval_H[3]-fex)
    plt.semilogy(xeval,err_l0,'o--', label='N = 5')
    plt.semilogy(xeval,err_l1,'o--', label='N = 10')
    plt.semilogy(xeval,err_l2,'o--', label='N = 15')
    plt.semilogy(xeval,err_l3,'o--', label='N = 20')
    plt.legend()

    plt.title("Absolute Error of Hermite Interpolation")
    plt.show()

def eval_hermite(xeval,xint,yint,ypint,N):

    ''' Evaluate all Lagrange polynomials'''

    lj = np.ones(N+1)
    for count in range(N+1):
       for jj in range(N+1):
           if (jj != count):
              lj[count] = lj[count]*(xeval - xint[jj])/(xint[count]-xint[jj])

    ''' Construct the l_j'(x_j)'''
    lpj = np.zeros(N+1)
#    lpj2 = np.ones(N+1)
    for count in range(N+1):
       for jj in range(N+1):
           if (jj != count):
#              lpj2[count] = lpj2[count]*(xint[count] - xint[jj])
              lpj[count] = lpj[count]+ 1./(xint[count] - xint[jj])
              

    yeval = 0.
    
    for jj in range(N+1):
       Qj = (1.-2.*(xeval-xint[jj])*lpj[jj])*lj[jj]**2
       Rj = (xeval-xint[jj])*lj[jj]**2
#       if (jj == 0):
#         print(Qj)
         
#         print(Rj)
#         print(Qj)
#         print(xeval)
 #        return
       yeval = yeval + yint[jj]*Qj+ypint[jj]*Rj
       
    return(yeval)
       
    

       
if __name__ == '__main__':
  # run the drivers only if this is called from the command line
  driver()        
