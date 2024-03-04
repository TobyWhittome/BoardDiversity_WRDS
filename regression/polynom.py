import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load your data
df = pd.read_excel('final_dataset.xlsx')

x = df['boardsize'].values
y = df['tobinsQ'].values

# Choose the degree of the polynomial
degree = 2  # Example for a quadratic model

# Fit the polynomial model
coefficients = np.polyfit(x, y, degree)

# Use the coefficients to create a polynomial function
polynomial = np.poly1d(coefficients)

# Generate a sequence of values within the observed range to plot the polynomial curve
x_sequence = np.linspace(x.min(), x.max(), 100)

# Calculate the corresponding y values for the polynomial curve
y_poly = polynomial(x_sequence)

# Plot the original data and the polynomial regression curve
plt.scatter(x, y, color='blue', label='Original data')
plt.plot(x_sequence, y_poly, color='red', label=f'Polynomial degree {degree}')
plt.title('Polynomial Regression with numpy')
plt.xlabel('Board Size')
plt.ylabel('Tobin\'s Q')
plt.legend()
plt.show()

# Calculating R-squared manually as np.polyfit does not directly provide it
y_pred = polynomial(x)
ss_res = np.sum((y - y_pred) ** 2)
ss_tot = np.sum((y - np.mean(y)) ** 2)
r_squared = 1 - (ss_res / ss_tot)
print(f'R-squared value: {r_squared}')
