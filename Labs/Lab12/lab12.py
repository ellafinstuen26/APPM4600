from gauss_legendre import *
from adaptive_quad import *
import matplotlib.pyplot as plt

def driver():
    a = 0.1
    b = 2
    f = lambda x: np.sin(1/x)
    tol = 1e-3
    n = 5

    T = adaptive_quad(a,b,f,tol,n,eval_composite_trap)
    S = adaptive_quad(a,b,f,tol,n,eval_composite_simpsons)
    G = adaptive_quad(a,b,f,tol,n,eval_gauss_quad)

    print(' Trap: ', T[0], T[2],'\n', 'Simp: ', S[0], S[2], '\n', 'Gauss: ',G[0], G[2])
driver()