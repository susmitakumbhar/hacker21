# Import the necessary libraries
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Generate some sample data
X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)  # Independent variable
y = np.array([2, 4, 5, 4, 5])             # Dependent variable

# Create and train the linear regression model
model = LinearRegression()
model.fit(X, y)

# Make predictions
X_new = np.array([6]).reshape(-1, 1)
y_pred = model.predict(X_new)

# Plot the data and the regression line
plt.scatter(X, y, label='Data points')
plt.plot(X, model.predict(X), color='red', label='Linear Regression')
plt.scatter(X_new, y_pred, color='green', marker='x', s=100, label='Prediction')
plt.legend()
plt.xlabel('X')
plt.ylabel('y')
plt.show()

# Print the coefficients of the linear regression model
print(f'Intercept: {model.intercept_}')
print(f'Coefficient: {model.coef_}')
