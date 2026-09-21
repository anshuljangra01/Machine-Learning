from sklearn.preprocessing import LabelEncoder
import pandas as pd
from pathlib import Path


csv_path = Path(__file__).parent.parent / 'label_data.csv'
df = pd.read_csv(csv_path, encoding='unicode_escape')

df_label = df.copy()
le = LabelEncoder()
df_label["Gender_Encoded"] = le.fit_transform(df_label["Gender"])
df_label["Passed_Encoded"] = le.fit_transform(df_label["Passed"])

print('\n Label Encoded DataFrame:')
# print(df_label[["Name", "Gender", "Gender_Encoded", "Passed", "Passed_Encoded"]].head())
print(df_label[["Name", "Gender", "Gender_Encoded", "Passed", "Passed_Encoded"]])

df_encoded = pd.get_dummies(df_label, columns= ["City"],dtype=int)

print("\n One Hot Encoded Data (city)")
print(df_encoded)