import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as la
import math

x1 = math.pi
x2 = 10**6
delta = np.array([10**i for i in range(-16,1)])

def func1(x, delta):
    y1 = np.cos(x+delta)-np.cos(x)
    return y1

def func2(x,delta):
    y2 = delta*(np.sin(x)-np.sin(x+delta))
    return y2
    
y11 = func1(x1,delta)
y12 = func2(x1,delta)
y13 = abs(y11-y12)

y21 = func1(x2,delta)
y22 = func2(x2,delta)
y23 = abs(y21-y22)

plt.plot(delta,y13)
plt.xlabel("delta")
plt.ylabel("difference between two expressions")
plt.title("Difference between the two expressions for x = pi")
plt.show()

plt.plot(delta,y23)
plt.xlabel("delta")
plt.ylabel("difference between two expressions")
plt.title("Difference between the two expressions for x = 10^6")
plt.show()
