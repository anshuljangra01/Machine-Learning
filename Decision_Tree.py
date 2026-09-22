from sklearn.tree import DecisionTreeClassifier

x= [
    [7,2],  # apple
    [8,3], # apple
    [9,8], # orange
    [10,9] # orange
]

y= [0,0,1,1] # 0 =Apple , 1 = Orange

model = DecisionTreeClassifier()

model.fit(x,y)

size= float(input("Enter the fruit size in cm: "))
shade= float(input("Enter the colour shade (1-10): "))

result  = model.predict([[size,shade]])[0]

if result == 0:
    print("This is likely an Apple")
else:    
    print("This is likely an Orange")
    