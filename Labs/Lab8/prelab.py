import numpy as np
import math

def driver():
    f = lambda x: x**2
    print(evaluate(0,2,f,4))



def evaluate(x0,x1,f,a):
    y0 = f(x0)
    y1 = f(x1)
  
    m = (y1 - y0) / (x1 - x0)
  
    
    return y0 + m * (a - x0)

driver()