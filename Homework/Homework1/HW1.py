import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as la
import math


def driver():

   n = 100
   #x = np.linspace(0,np.pi,n)
   # this is a function handle. You can use it to define
   #functions instead of using a subroutine like you
   # have to in a true low level language.
   x = np.arange(1.920,2.080,0.001)
   p1 = x**9-18*x**8+144*x**7-672*x**6+2016*x**5-4032*x**4+5376*x**3-4608*x**2+2304*x-512
   p2 = lambda x: (x-2)**9
   y1 = p1(x)
   y2 = p2(x)
   plt.plot(x,y1)
   plt.xlabel('x')
   plt.ylabel('y')
   plt.show()

   plt.plot(x,y2)
   plt.xlabel('x')
   plt.ylabel('y')
   plt.show()


   
   # evaluate the dot product of y and w

   # print the output
   print('the matrix multiplication is : ', c)
   return


def matrixMult(a,b):
   # Computes the dot product of the n x 1 vectors x and y
   c = np.matrix([[0,0], [0,0]])
   c[0,0] = a[0,0] * b[0,0] + a[0,1] * b[1,0]
   c[0,1] = a[0,0] * b[0,1] + a[0,1] * b[1,1]
   c[1,0] = a[1,0] * b[0,0] + a[1,1] * b[1,0]
   c[1,1] = a[1,0] * b[0,1] + a[1,1] * b[1,1]


   return c


driver()
