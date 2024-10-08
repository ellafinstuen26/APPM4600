import numpy as np
import math
import time
from numpy.linalg import inv 
from numpy.linalg import norm 

def driver():
    
    Nmax = 100
    tol = 1e-10

    # x = 1, y = 1
    print("(i):  x = 1, y = 1")
    x01 = np.array([1, 1])

    # t = time.time()
    # for j in range(50):
    #   [xstar,ier,its] =  Newton(x01,tol,Nmax)
    # elapsed = time.time()-t
    # print(xstar)
    # # F = evalF(xstar)
    # # print("Evaluation at solution: ",F)
    # print('Newton: the error message reads:',ier) 
    # print('Newton: took this many seconds:',elapsed/50)
    # print('Netwon: number of iterations is:',its)
     
    t = time.time()
    for j in range(20):
      [xstar,ier,its] = Broyden(x01, tol,Nmax)     
    elapsed = time.time()-t
    print(xstar)
    # F = evalF(xstar)
    # print("Evaluation at solution: ",F)
    print('Broyden: the error message reads:',ier)
    print('Broyden: took this many seconds:',elapsed/20)
    print('Broyden: number of iterations is:',its, "\n")

    t = time.time()
    for j in range(20):
      [xstar,ier,its] =  LazyNewton(x01,tol,Nmax)
    elapsed = time.time()-t
    print(xstar)
    print('Lazy Newton: the error message reads:',ier)
    print('Lazy Newton: took this many seconds:',elapsed/20)
    print('Lazy Newton: number of iterations is:',its,'\n')
     
        
    # x = 1, y = -1
    print("(ii):  x = 1, y = -1")
    x02 = np.array([1, -1])

    # t = time.time()
    # for j in range(50):
    #   [xstar,ier,its] =  Newton(x02,tol,Nmax)
    # elapsed = time.time()-t
    # print(xstar)
    # # F = evalF(xstar)
    # # print("Evaluation at solution: ",F)
    # print('Newton: the error message reads:',ier) 
    # print('Newton: took this many seconds:',elapsed/50)
    # print('Netwon: number of iterations is:',its)
     
    t = time.time()
    for j in range(20):
      [xstar,ier,its] = Broyden(x02, tol,Nmax)     
    elapsed = time.time()-t
    print(xstar)
    # F = evalF(xstar)
    # print("Evaluation at solution: ",F)
    print('Broyden: the error message reads:',ier)
    print('Broyden: took this many seconds:',elapsed/20)
    print('Broyden: number of iterations is:',its, "\n")

    t = time.time()
    for j in range(20):
      [xstar,ier,its] =  LazyNewton(x02,tol,Nmax)
    elapsed = time.time()-t
    print(xstar)
    print('Lazy Newton: the error message reads:',ier)
    print('Lazy Newton: took this many seconds:',elapsed/20)
    print('Lazy Newton: number of iterations is:',its,'\n')


    # x = 0, y = 0
    print("(iii):  x = 0, y = 0")
    print("Broyden: Error - Singular matrix, cannot take inverse")
    print("Lazy Newton: Error - Singular matrix, cannot take inverse")
    # x03 = np.array([0, 0])

    # t = time.time()
    # for j in range(50):
    #   [xstar,ier,its] =  Newton(x03,tol,Nmax)
    # elapsed = time.time()-t
    # print(xstar)
    #F = evalF(xstar)
    #print("Evaluation at solution: ",F)
    # print('Newton: the error message reads:',ier) 
    # print('Newton: took this many seconds:',elapsed/50)
    # print('Netwon: number of iterations is:',its)
     
    # t = time.time()
    # for j in range(20):
    #   [xstar,ier,its] = Broyden(x03, tol,Nmax)     
    # elapsed = time.time()-t
    # print(xstar)
    #F = evalF(xstar)
    #print("Evaluation at solution: ",F)
    # print('Broyden: the error message reads:',ier)
    # print('Broyden: took this many seconds:',elapsed/20)
    # print('Broyden: number of iterations is:',its)

    # t = time.time()
    # for j in range(20):
    #   [xstar,ier,its] =  LazyNewton(x03,tol,Nmax)
    # elapsed = time.time()-t
    # print(xstar)
    # print('Lazy Newton: the error message reads:',ier)
    # print('Lazy Newton: took this many seconds:',elapsed/20)
    # print('Lazy Newton: number of iterations is:',its,'\n')

def evalF(p): 

    F = np.zeros(2)
    x = p[0]
    y = p[1]

    F[0] = x**2+y**2-4
    F[1] = math.e**x+y-1
   
    return F
    
def evalJ(p): 
    x = p[0]
    y = p[1]
    J = np.array([[2*x,2*y],[math.e**x,1]])

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
           
def LazyNewton(x0,tol,Nmax):

    ''' Lazy Newton = use only the inverse of the Jacobian for initial guess'''
    ''' inputs: x0 = initial guess, tol = tolerance, Nmax = max its'''
    ''' Outputs: xstar= approx root, ier = error message, its = num its'''

    J = evalJ(x0)
    Jinv = inv(J)
    for its in range(Nmax):

       F = evalF(x0)
       x1 = x0 - Jinv.dot(F)
       
       if (norm(x1-x0) < tol):
           xstar = x1
           ier =0
           return[xstar, ier,its]
           
       x0 = x1
    
    xstar = x1
    ier = 1
    return[xstar,ier,its]   
    
def Broyden(x0,tol,Nmax):
    '''tol = desired accuracy
    Nmax = max number of iterations'''

    '''Sherman-Morrison 
   (A+xy^T)^{-1} = A^{-1}-1/p*(A^{-1}xy^TA^{-1})
    where p = 1+y^TA^{-1}Ax'''

    '''In Newton
    x_k+1 = xk -(G(x_k))^{-1}*F(x_k)'''


    '''In Broyden 
    x = [F(xk)-F(xk-1)-hat{G}_k-1(xk-xk-1)
    y = x_k-x_k-1/||x_k-x_k-1||^2'''

    ''' implemented as in equation (10.16) on page 650 of text'''
    
    '''initialize with 1 newton step'''
    
    A0 = evalJ(x0)

    v = evalF(x0)
    A = np.linalg.inv(A0)

    s = -A.dot(v)
    xk = x0+s
    for  its in range(Nmax):
       '''(save v from previous step)'''
       w = v
       ''' create new v'''
       v = evalF(xk)
       '''y_k = F(xk)-F(xk-1)'''
       y = v-w;                   
       '''-A_{k-1}^{-1}y_k'''
       z = -A.dot(y)
       ''' p = s_k^tA_{k-1}^{-1}y_k'''
       p = -np.dot(s,z)                 
       u = np.dot(s,A) 
       ''' A = A_k^{-1} via Morrison formula'''
       tmp = s+z
       tmp2 = np.outer(tmp,u)
       A = A+1./p*tmp2
       ''' -A_k^{-1}F(x_k)'''
       s = -A.dot(v)
       xk = xk+s
       if (norm(s)<tol):
          alpha = xk
          ier = 0
          return[alpha,ier,its]
    alpha = xk
    ier = 1
    return[alpha,ier,its]
     
        
if __name__ == '__main__':
    # run the drivers only if this is called from the command line
    driver()       
