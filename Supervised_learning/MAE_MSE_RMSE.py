# MAE - Mean Absolute Error  
# MSE - Mean Squared Error
#RMSE - Root Mean Squared Error 

from sklearn.metrics import mean_absolute_error, mean_squared_error , root_mean_squared_error
import numpy as np

# real score 
real_score = [90,60,80,100]
predicted_score = [85,70,70,95]

mae =mean_absolute_error(real_score,predicted_score)
mse =mean_squared_error(real_score,predicted_score)
rmse =root_mean_squared_error(real_score,predicted_score)

# rmse_2 = np.sqrt(mse)

print("MAE: On average off by: ",mae)
print("MSE: Squared Mistake Value: ",mse)
print("RMSE: Final Realistic error: ",rmse)