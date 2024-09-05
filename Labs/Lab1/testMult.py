import numpy as np
import numpy.linalg as la
import math

def driver():
    #n = 100
    #n = 10
    #x = np.linspace(0,np.pi,n)
    # this is a function handle. You can use it to define
    #functions instead of using a subroutine like you
    # have to in a true low level language.
    #f = lambda x: x**2 + 4*x + 2*np.exp(x)
    #g = lambda x: 6*x**3 + 2*np.sin(x)
    #y = f(x)
    #w = g(x)
    a = np.matrix([[1, 2], [3,4]])
    b = np.matrix([[1,1], [1,1]]) #creating the matrices for the multiplication

    #y = np.linspace(0,10,10)
    #w = np.linspace(5,20,10)

    # evaluate the dot product of y and w
    c = matrixMult(a,b) #calling function, might be called a process?
    # print the output
    print('the matrix multiplication is : ', c) #printing answer
    return

def matrixMult(a,b):
    # Computes the dot product of the n x 1 vectors x and y
    c = np.matrix([[0,0], [0,0]]) #creating new matrix for answer
    c[0,0] = a[0,0] * b[0,0] + a[0,1] * b[1,0]
    c[0,1] = a[0,0] * b[0,1] + a[0,1] * b[1,1]
    c[1,0] = a[1,0] * b[0,0] + a[1,1] * b[1,0] 
    c[1,1] = a[1,0] * b[0,1] + a[1,1] * b[1,1]

    return c

driver()