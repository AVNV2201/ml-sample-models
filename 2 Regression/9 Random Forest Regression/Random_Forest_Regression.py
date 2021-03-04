import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2].values

# preparing the randjom forest regression model
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor( n_estimators = 10000, random_state = 0 )
regressor.fit(x,y)

# plotting the graph
X_grid = np.arange(min(x), max(x), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(x, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Regression Model)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# predict the salary for 6.5
y_pred = regressor.predict([[6.5]])