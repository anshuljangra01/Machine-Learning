import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error
import numpy as np

data = pd.read_csv("data.csv")

x= data[['Hours']]
y= data['Score']

model = LinearRegression()
model.fit(x,y)

Predict_score = model.predict(x)

# Evaluate MAE, MSE, RMSE
mae = mean_absolute_error(y,Predict_score)
mse = mean_squared_error(y,Predict_score)
rmse = np.sqrt(mse)

# print result 
print("Mean Absolute Error (MAE): ",mae)
print("Mean Squared Error (MSE): ",mse)
print("Root Mean Squared Error (RMSE): ",rmse)


new_pred = model.predict([[7]]) 

print(f"Predicited score for 7 hours = {new_pred} ")