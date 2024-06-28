# -*- coding: utf-8 -*-
"""Diabetes Prediction(SVM).ipynb


Importing the Dependencies
"""

import numpy as np  #creating arrays
import pandas as pd #creating data frames (structures)
from sklearn.preprocessing import StandardScaler  #Standardizing the data
from sklearn.model_selection import train_test_split #Training and testing
from sklearn import svm  #Support vector machine
from sklearn.metrics import accuracy_score  #Calculate accuracy

"""Data Collection and Analysis

"""

#Loading the dataset (PIMA DIABETES DATASET) - KAGGLE
diabetes_dataset = pd.read_csv("diabetes.csv")

#printing first 5 rows of the dataset
diabetes_dataset.head()

# Number of rows and columns
diabetes_dataset.shape

# Statiscal measures of the data
diabetes_dataset.describe()

#Gives the count of the column based on values
diabetes_dataset['Outcome'].value_counts()

diabetes_dataset.groupby('Outcome').mean()

#separating data and labels

#axis = 1 --> column and axis = 0 --> row
X = diabetes_dataset.drop(columns='Outcome',axis=1)
Y = diabetes_dataset['Outcome']

print(X)

print(Y)

"""Data Standardizing"""

scaler = StandardScaler()

stan_data = scaler.fit_transform(X)

stan_data

X = stan_data

Y = diabetes_dataset['Outcome']

print(X)
print(Y)

"""Train Test Split"""

#stratify is mentioned as Y as we base the data based on 0 or 1
X_train,X_test,Y_train,Y_test = train_test_split(X,Y, test_size=0.2,stratify=Y,random_state=2)

print(X.shape,X_train.shape,X_test.shape)

"""Training the Model"""

classifier = svm.SVC(kernel='linear')

#training the svm to the dataset
classifier.fit(X_train,Y_train)

"""Model Evaluation"""

#accuracy score
X_train_prediction = classifier.predict(X_train)
X_train_prediction

training_data_accuracy = accuracy_score(X_train_prediction,Y_train)

print("Accuracy Score:",training_data_accuracy)

X_test_predict = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_predict,Y_test)
test_data_accuracy

"""Predictive System"""

input_data = (4,110,92,0,0,37.6,0.191,30)

#changing to a numpy array
ip_d_np = np.asarray(input_data)

#reshaping the data for one instance
#as the model was trained for 768 instances but we are giving only 1 input
#reshaping according to the input will give correct answer

ip_d_reshape = ip_d_np.reshape(1,-1)

#standardize the data
ip_d_stand = scaler.transform(ip_d_reshape)

print(ip_d_stand)



prediction = classifier.predict(ip_d_stand)
prediction

if prediction[0] == 0:
  print("Non Diabetic")
else:
  print("Diabetic ")

