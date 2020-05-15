#importing the library 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.metrics import classification_report,confusion_matrix

#loading the dataset
train=pd.read_csv("C:/Users/HP/Desktop/train (1).csv")
test=pd.read_csv("C:/Users/HP/Desktop/test (2).csv")
train=train.dropna()
test=test.dropna()
train.head()


X_train = np.array(train.iloc[:, :-1].values)
y_train = np.array(train.iloc[:, 1].values)
X_test = np.array(test.iloc[:, :-1].values)
y_test = np.array(test.iloc[:, 1].values)


#Huber Regressor
from sklearn.linear_model import  HuberRegressor
model = HuberRegressor()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = model.score(X_test, y_test)
plt.plot(X_train, model.predict(X_train), color='y')
plt.show()
print(accuracy)


