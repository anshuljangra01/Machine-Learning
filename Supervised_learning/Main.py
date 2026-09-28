from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np

# Input data: hours studied
X = np.array([1, 2, 3, 4, 5, 6, 7, 8]).reshape(-1, 1)

# Target data: marks scored
y = np.array([35, 40, 50, 55, 65, 70, 80, 85])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Display results
print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)
print("Predictions:", predictions)
print("Actual values:", y_test)
print("Mean squared error:", mean_squared_error(y_test, predictions))
print("R2 score:", r2_score(y_test, predictions))

# Predict marks for 9 hours of study
new_prediction = model.predict([[9]])
print("Predicted marks for 9 hours:", new_prediction[0])

# Create values for drawing the regression line
line_X = np.linspace(X.min(), 9, 100).reshape(-1, 1)
line_y = model.predict(line_X)

# Plot the data, regression line, and new prediction
plt.scatter(X, y, color="blue", label="Actual data")
plt.plot(line_X, line_y, color="red", label="Regression line")
plt.scatter(9, new_prediction[0], color="green", s=100, label="9-hour prediction")
plt.xlabel("Hours studied")
plt.ylabel("Marks scored")
plt.title("Hours Studied vs Marks Scored")
plt.legend()
plt.grid(True)
plt.show()