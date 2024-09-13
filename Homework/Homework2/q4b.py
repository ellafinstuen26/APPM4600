import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as la
import math
import random


def driver():
    x,y = getXY(1.2,0.1,15,0)

    plt.plot(x,y)
    plt.xlabel("x(theta)")
    plt.ylabel("y(theta)")
    plt.title("Parametric Curve for Given Parameters")
    plt.show()

    for i in range(10):
        p = random.uniform(0,2)
        x,y = getXY(i,0.05,2+i,p)
        plt.plot(x,y)
    
    plt.xlabel("x(theta)")
    plt.ylabel("y(theta)")
    plt.title("Parametric Curve for Given Parameters")
    plt.show()

    return

def getXY(R,deltaR,f,p):
    theta = np.linspace(0,2*math.pi,100)
    x = R*(1+deltaR*np.sin(f*theta+p))*np.cos(theta)
    y = R*(1+deltaR*np.sin(f*theta+p))*np.sin(theta)
    return x,y


driver()
