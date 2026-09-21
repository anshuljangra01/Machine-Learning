from sklearn.preprocessing import StandardScaler , MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder 
import pandas as pd

df = pd.read_csv('sample_data2.csv')

df_label = df.copy()

gender_le = LabelEncoder()
passed_le = LabelEncoder()

df_label["Gender_Encoded"] = gender_le.fit_transform(df_label["Gender"])
df_label["Passed_Encoded"] = passed_le.fit_transform(df_label["Passed"])

print('\n Label Encoded DataFrame:')
print(df_label[["Name", "Gender", "Gender_Encoded", "Passed", "Passed_Encoded"]])

scaler = StandardScaler()

scaler = scaler.fit_transform(df_label[['Gender_Encoded', 'Passed_Encoded']])

print("Standard Scaler Output: ")
# print(pd.DataFrame(scaler, columns=['Name', 'Gender','Passed']))

MinMax_scaler  =MinMaxScaler()
MinMax_scaler = MinMax_scaler.fit_transform(df_label[['Gender_Encoded', 'Passed_Encoded']])

print("MinMax scaler Output: ")
print(pd.DataFrame(MinMax_scaler, columns=[ 'Gender_Encoded', 'Passed_Encoded']))

x= df_label[['Gender_Encoded']]
y=df_label[ 'Passed_Encoded']

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2, random_state=42)

print("Training Data: ")


print(x_train)

print("Testing Data: ")
print(x_test)


print("Training Data: ")
print(y_train)

print("Testing Data: ")
print(y_test)