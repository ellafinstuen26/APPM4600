import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as la
import math

t = np.array([i*math.pi/30 for i in range(30)])
y = np.cos(t)
s = t*y
S = sum(s)

print("the sum is: ", S)

