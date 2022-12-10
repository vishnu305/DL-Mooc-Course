#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 23:59:25 2020

@author: pranavshastri
"""

#importing the library
import numpy as np
import pandas as pd
import tensorflow as tf

tf.__version__

# read the dataset & dividing it into independent & dependent variables
dataset = pd.read_csv('Churn_Modelling.csv')
X=dataset.iloc[:,3:13].values
y=dataset.iloc[:,13].values

print(X)
print(y)
#Encoding Categorical Data
# Label Encoding the "Gender" column
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X[:,2]=le.fit_transform(X[:,2])

print(X)
# One Hot Encoding the "Geography" column
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct=ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[1])],remainder='passthrough')
X = np.array(ct.fit_transform(X))

print(X)
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)


# Building the Neural Network

from keras.models import Sequential
from keras.layers import Dense

# initialise ANN
ann = Sequential()

# Adding Layers to ANN

ann.add(Dense(units=6,activation="relu"))

# Adding a hidden layer
ann.add(Dense(units=6,activation="relu"))

# Adding the Output Layer
ann.add(Dense(units=1,activation="sigmoid"))

#Compiling my ANN

ann.compile(optimizer="adam",loss="binary_crossentropy",metrics=['accuracy'])

# Train the ANN

ann.fit(X_train,y_train,batch_size=10,epochs=100)

# Predict using the ANN

y_predicted = ann.predict(X_test)
y_predicted= (y_predicted>0.5)

print(np.concatenate((y_predicted.reshape(len(y_predicted),1), y_test.reshape(len(y_test),1)),1))

# Connfusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score

cm = confusion_matrix(y_test, y_predicted)
print(cm)
accuracy_score(y_test, y_predicted)




