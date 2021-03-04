import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Data.csv')

x = dataset.iloc[:,:3].values
y = dataset.iloc[:,3].values

from sklearn.impute import SimpleImputer
obj = SimpleImputer()
obj = obj.fit(x[:, 1:3])
x[:, 1:3] = obj.transform(x[:, 1:3])

# =============================================================================
#dataset.iloc[:,1:3] = x[:,1:3] #changes the dataset
# =============================================================================

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer([('encoder', OneHotEncoder(), [0])], remainder='passthrough')
x = np.array(ct.fit_transform(x), dtype=np.float)

le = LabelEncoder()
y = le.fit_transform(y)

# test and train dataset

from sklearn.model_selection import train_test_split
x_train,x_test,y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=0)

# feature scaling 

from sklearn.preprocessing import StandardScaler
sc = StandardScaler() 
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

















