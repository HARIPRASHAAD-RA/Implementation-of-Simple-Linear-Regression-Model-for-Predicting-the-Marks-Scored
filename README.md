# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard Libraries.

2.Set variables for assigning dataset values.

3.Import linear regression from sklearn.

4.Assign the points for representing in the graph.

5.Predict the regression for marks by using the representation of the graph.

6.Compare the graphs and hence we obtained the linear regression for the given datas.
## Program:
```
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

dataset = pd.read_csv('student_scores.csv')
print(dataset.head())
print(dataset.tail())

dataset.info()

#assigning hours to x & scores to y
x = dataset.iloc[:,:-1].values
print(x)
y = dataset.iloc[:,-1].values
print(y)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(x_train,y_train)
y_pred = reg.predict(x_test)
print(y_pred)
print(y_test)

print(X_train.shape)
print(X_test.shape)
print(Y_test.shape)
print(Y_train.shape)

mse=mean_squared_error(y_test,y_pred) 
print('MSE = ',mse) 
mae=mean_absolute_error(y_test,y_pred) 
print('MAE = ',mae) 
rmse=np.sqrt(mse) 
print('RMSE = ',rmse)

plt.scatter(x_test,y_test,color="blue")
plt.plot(x_test,y_pred,color="silver")
plt.title('Test set(H vs S)')
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

a=np.array([[13]])
y_pred1 = reg.predict(a)
print(y_pred1)

/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: HARIPRASHAAD RA 
RegisterNumber: 212223040060  
*/
```

## Output:

![IMG-20250328-WA0019](https://github.com/user-attachments/assets/9946e849-af30-4655-98eb-6b5d549d7f32)
![IMG-20250328-WA0018](https://github.com/user-attachments/assets/2ceb29a3-dfcf-4e62-9368-50429754c251)
![IMG-20250328-WA0017](https://github.com/user-attachments/assets/044b8bfb-0c83-4e97-964e-d3119b73725e)
![IMG-20250328-WA0016](https://github.com/user-attachments/assets/bf8a863a-5605-4869-afa8-511a9cd119ea)
![IMG-20250328-WA0021](https://github.com/user-attachments/assets/9dad50aa-c913-4d5b-b867-0b8da9618284)
![IMG-20250328-WA0020](https://github.com/user-attachments/assets/b78804ab-6072-4fae-ab83-c4d08237854b)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
