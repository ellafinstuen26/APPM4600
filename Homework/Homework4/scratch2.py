import numpy as np
import matplotlib.pyplot as plt

# Define the function and its derivative
def f(x):
    return np.exp(3 * x) - 9 * x**2 * np.exp(2 * x) + 27 * x**4 * np.exp(x) - 27 * x**6

def f_prime(x):
    # Derivative of the function
    return 3 * np.exp(3 * x) - (18 * x * np.exp(2 * x) + 9 * x**2 * 2 * np.exp(2 * x)) + (108 * x**3 * np.exp(x) + 27 * x**4 * np.exp(x)) - 162 * x**5

# Generate x values and calculate y values for f'(x)
x_values = np.linspace(-2, 2, 400)
f_prime_values = f_prime(x_values)

# Plot the derivative function
plt.figure(figsize=(8, 6))
plt.plot(x_values, f_prime_values, label=r"$f'(x)$")
plt.axhline(1, color='red', linestyle='--', label="y = 1")
plt.title("Graph of $f'(x)$ and y=1 line")
plt.xlabel("x")
plt.ylabel("Slope (f'(x))")
plt.grid(True)
plt.axhline(0, color='black',linewidth=0.5)
plt.axvline(0, color='black',linewidth=0.5)
plt.legend()
plt.show()

# Find x values where f'(x) < 1
x_below_1 = x_values[abs(f_prime_values) < 1]
print(x_below_1)
