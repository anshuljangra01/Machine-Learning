from sklearn.metrics import accuracy_score , precision_score , recall_score, f1_score

# true answers
y_true = [1,0,1,1,0,1,0] 

y_pred= [1,0,1,0,0,1,1]

print("Accuracy: ", accuracy_score(y_true, y_pred))
print("Precison: ", precision_score(y_true, y_pred))
print("Recall: ", recall_score(y_true, y_pred))
print("F1 Score: ", f1_score(y_true, y_pred))