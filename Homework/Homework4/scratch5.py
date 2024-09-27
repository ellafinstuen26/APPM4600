import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

# Define the function and its derivative
def f(x):
    return x**6 - x - 1

def f_prime(x):
    return 6 * x**5 - 1

# Newton's Method with error tracking
def newtons_method(f, f_prime, x0, alpha, tol=1e-6, max_iter=100):
    x = x0
    errors = []
    for _ in range(max_iter):
        fx = f(x)
        fpx = f_prime(x)
        if abs(fpx) < 1e-10:
            break
        x_new = x - fx / fpx
        error = abs(x_new - alpha)
        if error > 0:  # Store only positive errors
            errors.append(error)
        if abs(x_new - x) < tol:
            break
        x = x_new
    return errors

# Secant Method with error tracking
def secant_method(f, x0, x1, alpha, tol=1e-6, max_iter=100):
    errors = []
    for _ in range(max_iter):
        f_x0 = f(x0)
        f_x1 = f(x1)
        if abs(f_x1 - f_x0) < 1e-10:
            break
        x2 = x1 - f_x1 * (x1 - x0) / (f_x1 - f_x0)
        error = abs(x2 - alpha)
        if error > 0:  # Store only positive errors
            errors.append(error)
        if abs(x2 - x1) < tol:
            break
        x0, x1 = x1, x2
    return errors

# Known root approximation (can be more precise)
alpha = 1.13472413875  # Approximate largest root of x^6 - x - 1
x0_newton = 2
x0_secant = 2
x1_secant = 1

# Apply Newton and Secant methods
errors_newton = newtons_method(f, f_prime, x0_newton, alpha)
errors_secant = secant_method(f, x0_secant, x1_secant, alpha)
print(errors_newton)

# Take logs of the errors
log_errors_newton_xk = np.log(errors_newton[:-1])
log_errors_newton_xk1 = np.log(errors_newton[1:])
log_errors_secant_xk = np.log(errors_secant[:-1])
log_errors_secant_xk1 = np.log(errors_secant[1:])

# Perform linear regression to calculate the slope
slope_newton, intercept, r_value, p_value, std_err = linregress(log_errors_newton_xk, log_errors_newton_xk1)
slope_secant, intercept, r_value, p_value, std_err = linregress(log_errors_secant_xk, log_errors_secant_xk1)

# Print the slopes (order of convergence)
print(f"Slope (Order of Convergence) for Newton's Method: {slope_newton}")
print(f"Slope (Order of Convergence) for Secant Method: {slope_secant}")

# Plot |x_k - α| vs |x_{k+1} - α| on log-log axes
plt.figure(figsize=(10, 6))
plt.loglog(errors_newton[:-1], errors_newton[1:], label="Newton's Method", marker='o', color="blue")
plt.loglog(errors_secant[:-1], errors_secant[1:], label="Secant Method", marker='o', color="pink")
plt.xlabel(r'$|x_k - \alpha|$')
plt.ylabel(r'$|x_{k+1} - \alpha|$')
plt.title('Convergence of Newton and Secant Methods (log-log scale)')
plt.legend()
plt.grid(True)
plt.show()
