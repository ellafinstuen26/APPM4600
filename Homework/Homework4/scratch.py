# import libraries
import math
import scipy
from scipy import special
import numpy as np
import matplotlib.pyplot as plt
        
def driver():
    x = np.arange(3,5,.05)
    f = lambda x: np.exp(3*x)-27*x**6+27*x**4*np.exp(x)-9*x**2*np.exp(2*x)
    fp = lambda x: 3*np.exp(3*x)-162*x**5+ (27*x**4*np.exp(x) +108*x**3*np.exp(x)) - (18*x**2*np.exp(2*x)+ 18*x*np.exp(2*x))

    y = f(x)
    y_prime = fp(x)

    
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.axhline(0, color = 'black', linewidth=.9)
    plt.axvline(0, color = 'black', linewidth=.9)
    plt.plot(x,y)
    plt.title("f(x) on [3,5]")
    plt.show()

    plt.xlabel("x")
    plt.ylabel("f'(x)")
    plt.axhline(0, color = 'black', linewidth=.9)
    plt.axvline(0, color = 'black', linewidth=.9)
    plt.plot(x,y_prime)
    plt.title("f'(x) on [3,5]")
    plt.show()




   
        
driver()
