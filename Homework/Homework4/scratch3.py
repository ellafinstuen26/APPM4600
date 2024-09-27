import numpy as np

# Define the function f(x) = x^6 - x - 1
def f(x):
    return x**6 - x - 1

# Secant method implementation
def secant_method(f, x0, x1, tol=1e-6, max_iter=100):
    for i in range(max_iter):
        f_x0 = f(x0)
        f_x1 = f(x1)
        
        if abs(f_x1 - f_x0) < 1e-10:  # Avoid division by a very small number
            print("Small difference in function values, stopping iteration.")
            break
        
        # Secant method formula
        x2 = x1 - f_x1 * (x1 - x0) / (f_x1 - f_x0)
        
        if abs(x2 - x1) < tol:  # Check for convergence
            return x2
        
        x0, x1 = x1, x2
    
    return x2

# Initial guesses and applying the secant method
x0 = 2
x1 = 1
root = secant_method(f, x0, x1)
print(root)
