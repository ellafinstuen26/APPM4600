import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as la
import math

def driver():
    x = np.arange(-5,10,.1)
    f = lambda x: x-4*np.sin(2*x)-3
    y = f(x)

    
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axhline(0, color = 'black', linewidth=.9)
    plt.axvline(0, color = 'black', linewidth=.9)
    plt.plot(x,y)
    plt.title("f(x) plot that shows all zero crossings")
    plt.show()


    


driver()