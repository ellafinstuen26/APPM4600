import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

def driver():

    f = lambda x: np.sin(x)
    m = lambda x: x-x**3/6+x**5/120

    p1 = lambda x: (x-7*x**3/60)/(1+x**2/20)
    p2= lambda x: (x)/(1+x**2/6+7*x**4/360)


    a = 0
    b = 5
    Neval = 100
    xeval = np.linspace(a,b,Neval+1)

    fex = f(xeval)
    
    err_m = abs(m(xeval)-fex)
    err_p1 = abs(p1(xeval)-fex)
    err_p2 = abs(p2(xeval)-fex)


    plt.figure() 
    plt.semilogy(xeval,err_m,'o--', label='Macluarin Series')
    plt.semilogy(xeval,err_p1,'o--', label='$P_3^3(x) = P_2^4(x)$')
    plt.semilogy(xeval,err_p2,'o--', label='$P_4^2(x)$')
    
    plt.legend()

    plt.title("Absolute Error of Various Approximations of $f(x) = sin(x)$")
    plt.show()

       

driver()        
