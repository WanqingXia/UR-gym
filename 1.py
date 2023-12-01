import numpy as np
import matplotlib.pyplot as plt

# Define the function e^(10x)
def exponential_function(x):
    return np.exp(2 * x)

# Define the modified exponential function with logarithmic transformation
def modified_exponential_function(x, a=2, b=2):
    return 1 - np.exp(-a * np.log(1 + b * x))


# Create an array of x values from 0 to 1
x = np.linspace(0, 1, 400)

# Calculate y values based on the exponential function
y = exponential_function(x)

# Create the plot
plt.figure(figsize=(8, 6))
plt.plot(x, y, label='f(x) = e^(10x)')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Graph of f(x) = e^(10x)')
plt.legend()
plt.grid(True)
plt.show()
