# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 21:36:50 2021

@author: abc
"""

"""

Application of Autoencoders - Anomaly Detection

"""


import pandas as pd

#read our dataset
df = pd.read_csv('anomaly.csv')
print(df.head())

#To see how the data is spread between Good and Bad
print(df.groupby('Quality')['Quality'].count())

#drop unneccecary data
df.drop(['Date'], axis=1, inplace=True)

#If there are missing entries, drop them
df.dropna(inplace=True, axis=1)
print(df.head())

#Convert non-numeric to numeric
df.Quality[df.Quality == 'Good'] = 1
df.Quality[df.Quality == 'Bad'] = 2

#All good to be True for good data points
good_mask = df['Quality'] == 1 

#All values false for good data points
bad_mask = df['Quality'] == 2

print(good_mask.head())
print(bad_mask.head())

#we drop quallity because we only need power and detector
df.drop('Quality', axis=1, inplace=True)

df_good = df[good_mask]
df_bad = df[bad_mask]
print(df_bad.head())

#Sanity check to see if we have same number of good and bad datapoints
print(f"Good count: {len(df_good)}")
print(f"Bad count: {len(df_bad)}")

#This is the feature vector that goes to the neural net
x_good = df_good.values
x_bad = df_bad.values

from sklearn.model_selection import train_test_split

x_good_train, x_good_test = train_test_split(x_good, test_size=0.25, random_state=42)

print(f"Good train count: {len(x_good_train)}")
print(f"Good test count: {len(x_good_test)}")


#############################################################


#Define the autoencoder model
#Since  we are dealing with numeric values we can use only Dense Layers

from sklearn import metrics
import numpy as np
import pandas as pd

#Create model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


#create a model
model = Sequential()
model.add(Dense(10, input_dim=x_good.shape[1], activation='relu'))
model.add(Dense(3, activation="relu"))
model.add(Dense(10, activation='relu'))
model.add(Dense(x_good.shape[1]))
model.compile(loss='mean_squared_error', optimizer='adam')
model.summary()


#Fit the model
model.fit(x_good_train, x_good_train, verbose=1, epochs=100)

#Predict our model
pred = model.predict(x_good_test)
score1 = np.sqrt(metrics.mean_squared_error(pred, x_good_test))

pred = model.predict(x_good)
score2 = np.sqrt(metrics.mean_squared_error(pred, x_good))

pred = model.predict(x_bad)
score3 = np.sqrt(metrics.mean_squared_error(pred, x_bad))

print(f"Insample Good Score (RMSE): {score1}".format(score1))
print(f"Out of Sample Good Score (RMSE): {score2}")
print(f"Bad sample Score (RMSE) : {score3}")



