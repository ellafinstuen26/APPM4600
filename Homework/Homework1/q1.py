import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as la
import math

x = np.arange(1.920,2.080,0.001)
p1 = lambda x: x**9-18*x**8+144*x**7-672*x**6+2016*x**5-4032*x**4+5376*x**3-4608*x**2+2304*x-512
p2 = lambda x: (x-2)**9
y1 = p1(x)
y2 = p2(x)
plt.plot(x,y1)
plt.title('X vs P(x) evaluating P via coefficients')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

plt.plot(x,y2)
plt.title('X vs P(x) evaluating P via the expression (x-2)^9')
plt.xlabel('x')
plt.ylabel('y')
plt.show()