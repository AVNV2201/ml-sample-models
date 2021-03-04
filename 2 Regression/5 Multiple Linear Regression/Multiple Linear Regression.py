#import library
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#imprt dataset
dataset = pd.read_csv('50_Startups.csv')

#get matrices
x = dataset.iloc[:,:4].values
y = dataset.iloc[:,4].values

#preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer([('encoder', OneHotEncoder(), [3])], remainder='passthrough')
x = np.array(ct.fit_transform(x), dtype=np.float)

# get rid of dummy variable trap
x = x[:,1:] 

#splitting dataset
from sklearn.model_selection import train_test_split
x_train,x_test,y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=0)

#prepare model
from sklearn.linear_model import LinearRegression
regresser = LinearRegression()
regresser.fit( x_train, y_train )

# predict values
y_pred = regresser.predict( x_test )

#adding column of ones in the beginning
x = np.append( arr = np.ones((50,1)).astype(int), values = x, axis = 1 )

#backward elimination hardocded
"""
import statsmodels.api as sm
x_opt = x[:, [ 0,1,2,3,4,5 ] ]
regresser_OLS = sm.OLS( endog = y, exog = x_opt ).fit()
regresser_OLS.summary()
x_opt = x[:, [ 0,1,3,4,5 ] ]
regresser_OLS = sm.OLS( endog = y, exog = x_opt ).fit()
regresser_OLS.summary()
x_opt = x[:, [ 0,3,4,5 ] ]
regresser_OLS = sm.OLS( endog = y, exog = x_opt ).fit()
regresser_OLS.summary()
x_opt = x[:, [ 0,3,5 ] ]
regresser_OLS = sm.OLS( endog = y, exog = x_opt ).fit()
regresser_OLS.summary()
x_opt = x[:, [ 0,3 ] ]
regresser_OLS = sm.OLS( endog = y, exog = x_opt ).fit()
regresser_OLS.summary()
"""

# code for backward elimination
import statsmodels.api as sm
sl = 0.05
x_opt = x[:,[0,1,2,3,4,5]]
for i in range(0,6):
    regresser_ols = sm.OLS(endog=y,exog=x_opt).fit()
    max_p_val = max(regresser_ols.pvalues).astype(float)
    if( max_p_val > sl ):
        for j in range(0,6-i):
            if( regresser_ols.pvalues[j].astype(float) == max_p_val ):
                x_opt = np.delete(x_opt,j,1)
    else:
        break

















