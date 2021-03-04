# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#get the dataset
dataset = pd.read_csv('Position_Salaries.csv')

x = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2].values

# feature scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
sc_y = StandardScaler()
x = sc_x.fit_transform(x)
y = sc_y.fit_transform(y.reshape(-1,1))

# prepare svr model
from sklearn.svm import SVR
regresser = SVR( kernel = 'rbf' )
regresser.fit( x, y )

# plot the graph for svr model
plt.scatter(x,y)
plt.plot(x,regresser.predict(x) )
plt.title('svr model')
plt.xlabel('positions')
plt.ylabel('salaries')
plt.show()

# predicting th salary at level 6.5
y_pred = sc_y.inverse_transform(regresser.predict( sc_x.transform([[6.5]]) ))











