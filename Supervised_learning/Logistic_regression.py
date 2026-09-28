from sklearn.linear_model import LogisticRegression

x= [[1], [2], [3], [4],[5]] # hours studied
y= [[0], [0],[1], [1],[1]] # passed (1) or failed (0)

model = LogisticRegression()

model.fit(x,y)
hour = float(input("Enter how many hours you studies = "))

result = model.predict([[hour]])[0]

if result == 1:
    print(f"Based on hours {hour}, you are likely to Pass")
else:    
    print(f"Based on hours {hour}, you are likely to Fail")