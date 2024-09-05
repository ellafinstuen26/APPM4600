import numpy as np
import matplotlib.pyplot as plt
# def func(x):
#     return x**2

# print(func(9))

x = np.linspace(0,10,50)
y = np.arange(0,15,50)
print('the first three entries of x are ',x[0:3])
w = 10**(-np.linspace(1,10,10))
print(w)
x = np.arange(0,10,1)
s = 3*w
plt.semilogy(x,w)
plt.semilogy(x,s)
plt.xlabel('x')
plt.ylabel('w')
plt.show()
