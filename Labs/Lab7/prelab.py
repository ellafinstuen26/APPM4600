#import mypkg.my2DPlotB
import numpy as np
import math
from numpy.linalg import inv 

def driver():
    x = np.linspace(-1,-1,1000)
    F = np.array([1/(1+(10*j)**2) for j in x])
    a = solve(x,F)
    print(a)

def solve(x,F):
    n  = len(x)
    V = np.zeros((n,n))

    for i in range(n):
        for j in range(n):
            V[i][j] = x[i]**j

    print(V)


    Vinv = inv(V)
    a = Vinv.dot(F)
    return a

driver()
