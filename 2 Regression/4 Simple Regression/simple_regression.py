import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

# Simple Linear Regression Model
from sklearn.linear_model import LinearRegression
regresser = LinearRegression()
regresser.fit(X_train, y_train)

# predictions  of the X_test
y_pred = regresser.predict(X_test)

# plotting graph of training data
plt.scatter(X_train, y_train, color = 'red' )
plt.plot(X_train, regresser.predict(X_train), color = 'blue' )
plt.title('Simple Linear Regression Model ( Training Set )')
plt.xlabel('Year of experience')
plt.ylabel('Salary')
img = plt.show()

# plot of test set
plt.scatter(X_test, y_test, color = 'red' )
plt.plot(X_train, regresser.predict(X_train), color = 'blue' )
plt.title('Simple Linear Regression Model ( Test Set )')
plt.xlabel('Year of experience')
plt.ylabel('Salary')
img = plt.show()