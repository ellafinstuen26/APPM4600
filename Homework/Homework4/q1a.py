import math
import scipy
from scipy import special
import numpy as np
import matplotlib.pyplot as plt

def driver():
    d = 2*math.sqrt(5184000*0.138*10**(-6))
    x = np.arange(0,5,.1)
    f = lambda x: 35*scipy.special.erf(x/d)-15
    y = f(x)

    
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.axhline(0, color = 'black', linewidth=.9)
    plt.axvline(0, color = 'black', linewidth=.9)
    plt.plot(x,y)
    plt.title("f(x) on [0,5] to see temperature behavior at 60 days")
    plt.show()







driver()
