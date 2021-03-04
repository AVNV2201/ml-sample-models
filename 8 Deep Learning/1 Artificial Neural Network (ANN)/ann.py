# Artificial Neural Networks

#DATA PREPROCESSING3

#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:,3:-1].values
y = dataset.iloc[:,-1].values

# encode the categorical columns
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
le = LabelEncoder()
X[:,2] = le.fit_transform(X[:,2])
le2 = LabelEncoder()
X[:,1] = le2.fit_transform(X[:,1])
ct = ColumnTransformer([('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = np.array(ct.fit_transform(X), dtype=np.float)
#to avoid dummy variable trap, remove the first column
X = X[:,1:]

# splitting the data into training and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=0 )

# much necessary feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# the main part

import keras
from keras.models import Sequential
from keras.layers import Dense

# create neural network object
classifier = Sequential()

# adding input layer and first hiddden layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))

# adding second hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))

# adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# compiling the neural network
classifier.compile( optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'] )

# fitting dataset to the model
classifier.fit( X_train, y_train, batch_size=10, epochs = 100 )

# predicting the test set
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5 )

# checking the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix( y_test, y_pred )







