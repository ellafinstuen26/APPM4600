import matplotlib.pyplot as plt
import numpy as np
import math
from numpy.linalg import inv
from numpy.linalg import norm

def driver():

    f = lambda x: 1./(1.+x**2)
    fp = lambda x: -2*x/(1.+x**2)**2
    a = -5
    b = 5
    NS = [5,10,15,20]
    Neval = 1000
    xeval = np.linspace(a,b,Neval+1)
    yeval_l = []
    fex = f(xeval)

    for n in range(4):
       xint = np.array([5*np.cos((2*i-1)*np.pi/(2*NS[n])) for i in range(1,NS[n]+1)][::-1])

       yint = f(xint)

       (M,C,D) = create_clamped_spline(yint,xint,NS[n]-1,fp)
       yeval = eval_cubic_spline(xeval,Neval,xint,NS[n]-1,M,C,D)

       yeval_l.append(yeval)
       

    #xeval = np.linspace(xint[0],xint[N],Neval+1)
    # '''evaluate f at the evaluation points'''
    # fex = f(xeval)




    plt.figure()    
    plt.plot(xeval,fex,'rs-', label='Exact function')
    plt.plot(xeval,yeval_l[0],'o--', label='N = 5') 
    plt.plot(xeval,yeval_l[1],'o--', label='N = 10') 
    plt.plot(xeval,yeval_l[2],'o--', label='N = 15') 
    plt.plot(xeval,yeval_l[3],'o--', label='N = 20') 
    plt.title("Clamped Cubic Spline for various N and Exact function")
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

    plt.title("Absolute Error of Clamped Cubic Spline")
    plt.show()



def create_clamped_spline(yint,xint,N,fp):

    # create the right hand side for the linear system
    b = np.zeros(N+1)
    h = np.zeros(N)
    
    for i in range(1,N):
        hi = xint[i]-xint[i-1]
        hip = xint[i+1]-xint[i]

        b[i] = (yint[i+1]-yint[i])/hip - (yint[i]-yint[i-1])/hi
        h[i-1] = hi
        h[i] = hip
   
    h0 = xint[1]-xint[0]
    hN = xint[N]-xint[N-1]


    b[0] = (yint[1]-yint[0])/h0 - fp(xint[0])
    
    b[-1] = fp(xint[-1])-(yint[-1]-yint[-2])/hN

    # create the matrix A so you can solve for the M values
    A = np.zeros((N+1,N+1))
    A[0,0] = h0/3
    A[0,1] = h0/6
    for i in range(1,N):
        A[i,i-1] = h[i-1] / 6.0
        A[i,i] = (h[i-1] + h[i]) / 3.0
        A[i,i+1] = h[i] / 6.0

    A[N,N] = hN/3
    A[N,N-1] = hN/6
    

    # print(A)
    # print(b)

    # Invert A
    Ainv = inv(A)
    
    # solver for M
    M = Ainv @ b
    
    # Create the linear coefficients
    C = np.zeros(N)
    D = np.zeros(N)

    for j in range(N):
        C[j] = (yint[j]) / h[j] - h[j] * (M[j]) / 6.0
        D[j] = yint[j+1]/h[j] - h[j]*M[j+1]/6.0
    return(M,C,D)



def eval_local_spline(xeval,xi,xip,Mi,Mip,C,D):
    
    hi = xip-xi
    yeval = Mi*(xip-xeval)**3/(6*hi) + Mip*(xeval-xi)**3/(6*hi) + C*(xip-xeval) + D*(xeval-xi)
    return yeval


def eval_cubic_spline(xeval,Neval,xint,Nint,M,C,D):
    yeval = np.zeros(Neval+1)
    for j in range(Nint):
    # find indices of xeval in interval (xint(jint),xint(jint+1)), let ind denote the indices in the intervals
        atmp = xint[j]
        btmp= xint[j+1]

        #find indices of values of xeval in the interval
        ind= np.where((xeval >= atmp) & (xeval <= btmp))
        xloc = xeval[ind]

        # evaluate the spline
        yloc = eval_local_spline(xloc,atmp,btmp,M[j],M[j+1],C[j],D[j])

        # copy into yeval
        yeval[ind] = yloc
    return(yeval)

driver()