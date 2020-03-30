# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 21:57:42 2020

@title: Hydrogen > Stock market prediction using LSTM (RL - Deep Learning) as AI engine
@author: Leonardo Pedroso dos Santos - 
@author: Department of Computer Science - University of Oxford - United Kingdom

"""
import math
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Preprocessing and feature selection
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split

# Confusion Matrix
from sklearn.metrics import confusion_matrix

# Neural Network - LSTM (long short term memory)
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

# Support Vector Machine
from sklearn.svm import SVC

# Genetic Algorithm - NEAT


####################################### Step 1: Read Data #######################################  





####################################### Step 2: Data Preprocessing #######################################    

# SVM: Set up train and test databases
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)

# SVM: Feature scaling
sc = StandardScaler()
x_train = sc.fit_transform(X_train)
x_test = sc.transform(X_test)
   



        

####################################### Step 3: Set up AI model #######################################

# SVM 
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train, y_train)

# LSTM
model = Sequential() # Initialize NN
model.add(LSTM(50, return_sequences=True, input_shape= (x_train.shape[1], 1))) # Input Layer
model.add(LSTM(50, return_sequences=False)) # Hidden Layer
model.add(Dense(25)) # Hidden Layer
model.add(Dense(1)) # Output layer
    
model.compile(optimizer='adam', loss='mean_squared_error') # Compile model
model.fit(x_train, y_train, batch_size=1, epochs=10) 
    

# NEAT



####################################### Step 4: Run Models #######################################


# SVM
y_pred = classifier.predict(X_test)



    
####################################### Step 5: Check results and save models #######################################

# Pickle 