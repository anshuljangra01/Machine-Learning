from sklearn.preprocessing import StandardScaler ,MinMaxScaler
from sklearn.model_selection import train_test_split       
import pandas as pd


data = {
    'Study_Hours': [1, 2, 3, 4, 5, 6, 7], 
    'Marks_Scored': [35, 40, 50, 55, 65, 70, 80]
}

df= pd.DataFrame(data)
# standard scaler

Standard_Scaler= StandardScaler()
# Standard_Scaler= Standard_Scaler.fit_transform(df[['Study_Hours', 'Marks_Scored']])
Standard_Scaler= Standard_Scaler.fit_transform(df)

print("standard scaler Output: ")
print(pd.DataFrame(Standard_Scaler, columns=['Study_Hours', 'Marks_Scored']))

minmax_scaler= MinMaxScaler()
minmax_scaler= minmax_scaler.fit_transform(df)
print("MinMax Scaler Output: ")
print(pd.DataFrame(minmax_scaler, columns=['Study_Hours', 'Marks_Scored']))

x= df[['Study_Hours']]
y= df[['Marks_Scored']]
x_train, x_test, y_train, y_test= train_test_split(x,y,test_size=0.2, random_state=42)

print("Training Data: ")
print(x_train)

print("Test Data: ")
print(x_test)


print("Training Data: ")
print(y_train)

print("Test Data: ")
print(y_test)