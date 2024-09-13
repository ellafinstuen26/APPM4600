import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as la
import math
import random

def driver():

    x  = 9.999999995000000*10**(-10)
    y = eval(x)
    print(y)

    t = taylor(x)
    print(f"{t:.17}")

    return

def eval(x):
    y = math.e**x
    return y-1

def taylor(x):
    y = x + x**2/2
    return y

driver()

