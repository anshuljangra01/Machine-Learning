from sklearn.linear_model import LinearRegression
# import pandas as pd

x= [[1], [2], [3], [4],[5]]
y= [[35], [40], [50], [55],[65]]
model = LinearRegression()

model.fit(x,y)

hours = float(input("Enter the number of hours studied: "))
Predicted_marks = model.predict([[hours]])

print(f"Based on your hours {hours} you may score around {Predicted_marks}")


