import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#import dataset
dataset = pd.read_csv("Position_Salaries.csv")
x = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2].values

# linear model
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(x, y)

# preparing model
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures( degree = 2 )
x_poly = poly_reg.fit_transform(x)
poly_reg.fit(x_poly, y)
lin_reg2 = LinearRegression()
lin_reg2.fit( x_poly, y)

#plot simple linear regression 
plt.scatter(x,y)
plt.plot( x, lin_reg.predict(x) )
plt.title( 'simple linear regression demo ')
plt.xlabel( 'level' )
plt.ylabel('salaries')
plt.show()

#plot for polynomial regression 
x_grid = np.arange( min(x), max(x), 0.1 )
x_grid = x_grid.reshape(len(x_grid),1)
plt.scatter(x,y)
plt.plot(x_grid, lin_reg2.predict( poly_reg.fit_transform(x_grid) ) )
plt.title('polynomial regression demo ' )
plt.xlabel('position')
plt.ylabel('salaries')
plt.show()

#predict the salary through linear regression
lin_reg.predict([[6.5]])

#predicting the salary through polynomial regression
lin_reg2.predict( poly_reg.fit_transform([[6.5]]))










