import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# step no 2
x = np.array([[25,50000,3],[35,90000,2],[40,60000,5],[45, 80000,3],[20,30000,2],[55,120000,4],
              [28,40000,1],[32,100000,3],[38,75000,2]])
y = np.array([0,1,1,0,1,0,1,0,1])
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=42)
model = DecisionTreeClassifier()
model.fit(x_train,y_train)
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)
# user input
age = float(input("Enter age:"))
income = float(input("Enter Income:"))
education = float(input("Enter Education level:"))
user_input = np.array([[age, income, education]])
prediction = model.predict(user_input)
if prediction[0] == 1:
    print("The user is likely to purchase a smartphone")
else:
    print("The user is unlikely to purchase a smatphone")