# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 15:19:44 2020

@author: Leonardo
"""

# Support libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math

# Ploting - Testing
#import plotly.io as pio
#import plotly.graph_objs as go
#import plotly.offline as py
#from plotly.offline import plot, iplot

# Preprocessing libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer

# Machine Learning Models
from tensorflow.keras import backend
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

# Data Analysis 
# TO DO >> Create a classification 'buy', 'hold', 'short' given some % margin
# TO DO >> Data set for fundamentalist analysis

#py.init_notebook_mode(connected = True)
#plt.style.use('fivethirtyeight')

# Establish a stock trend at our series : TO DO
def stockTrend(series):
    
    return ['buy','short','hold']


def plotting():
    
    print("plow...")


def advancedPlotting():
    
    print("Plow!")


# Calculates the Relative Strength Index
def calulateRSI(series, period):
    
       delta = series['Close'].diff() # Get the difference in price from previous step

       # Calculate the variation between "close" values
       delta = delta[1:] # Get rid of the first row, which is NaN since it did not have a previous

       # Make the positive gains (up) and negative gains (down) Series
       up, down = delta.copy(), delta.copy()

       up[up < 0.0] = 0.0

       down[down > 0.0] = 0.0

       # Calculate the EWMA -  Exponential Weighted Moving Average

       roll_up1 = up.ewm(com=(period-1), min_periods=period).mean() # EWMA up

       roll_down1 = down.abs().ewm(com=(period-1), min_periods=period).mean() # EWMA down

       # Calculate the RSI based on EWMA
       RS1 = roll_up1 / roll_down1

       RSI1 = 100.0 - (100.0 / (1.0 + RS1))

       return RSI1    


############################################ Import Data Frame ############################################

df = pd.read_csv(r'C:\Users\Leonardo\Desktop\Oxford\Stock Market AI\Data\PETR4_SA.csv', parse_dates = ['Date']) # parse_dates = ['Date'] >> Force this column to be a date. 

df.dropna(inplace=True) # Remove NaN lines from stock market holydays

df_close = df.filter(['Close'])

dataset_close = df_close.values

# Testing: 1
#df.plot(x='Date', y='Adj Close', kind='line')
#plt.show()

#df.sort_values(by='Date', inplace=True, ascending=False) # Sorting for last recent data

################################## Variables Settup ########################################################

forecast_days = 7
training_data_len = math.ceil( len(dataset_close) * 0.8 ) # 80% for training
current_price = 0
n_stocks = 0
wallet = n_stocks * current_price

################################# Technical Investments Indexes ############################################

# Close vs Adj Close 
# Closing price of a stock is the price of that stock at the close of the trading day. 
# Adjusted closing price is a more complex analysis that uses the closing price as a starting point, but it takes into account factors such as dividends, stock splits and new stock offerings to determine a value

# Moving average 7 days
df['Ma7d'] = df['Close'].rolling(window=7, min_periods = 0).mean() # Caution: NaN data!!!

# Moving average 21 days
df['Ma21d'] = df['Close'].rolling(window=21, min_periods = 0).mean() # Caution: NaN data!!

# Deltas
df['Delta'] = df['Close'].diff()

# RSI - Relative Strength Index (Range 0 to 100)
# Calculus: 100 - 100 / (1 + RS)
# RS - Relative Strength >> Average gain of last 14 trading days / Average loss of last 14 trading days
df['Rsi'] = calulateRSI(df,14)

################################ Data Preprocessing and Training data sets ######################################################

# NaN treatment  >>> Bound with other variables for a better LSTM
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(df[['Delta','Rsi']])
df[['Delta','Rsi']] = imputer.transform(df[['Delta','Rsi']])

# Scale the data 
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset_close)

# Create traing data set with 80% of our original dataset
train_dataset = scaled_data[0:training_data_len, :]

#split in x_train and y_train
x_train = []
y_train = []

# 60 for 60 days
for i in range(60, len(train_dataset)):

    x_train.append(train_dataset[i-60:i, 0])
    y_train.append(train_dataset[i, 0])
    
    if i <= 60:
        print(x_train)
        print(y_train)
        print()
        
# Convert x_train and y_train to numpuy arrays
x_train, y_train = np.array(x_train), np.array(y_train)

# Reshape the train dataset because LSTM expect 3D data
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1)) # >> 1 because we are using just the "close"

################################ Machine Learning Models #################################################

#### LSTM ###
lstm = Sequential()
lstm.add(LSTM(50, return_sequences=True, input_shape= (x_train.shape[1], 1))) # Input layer
lstm.add(LSTM(50, return_sequences=False)) # Hidden
lstm.add(Dense(25)) # Hidden
lstm.add(Dense(1)) # Output

# Compile the model
lstm.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
lstm.fit(x_train, y_train, batch_size= 1, epochs= 10) # >> Change here later

# Create new array with scaled values with testing data
test_data = scaled_data[training_data_len - 60:, :]

# Create x_test, y_test
x_test = []
y_test = dataset_close[training_data_len:, :]

for i in range(60, len(test_data)):

    x_test.append(test_data[i-60:i, 0])
        
# Covert the data to numpy array 
x_test = np.array(x_test)

# Reshape the data
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# Get the model predictions for price values
predictions = lstm.predict(x_test)
predictions = scaler.inverse_transform(predictions) # Transform in readable data

# Get the root mean squared error (RMSE)
rmse = np.sqrt(np.mean(predictions - y_test) ** 2)
print(rmse)

# Plot data!
train = df_close[:training_data_len]
valid = df_close[training_data_len:]
valid['Predictions'] = predictions

# Visualize the model 
plt.figure(figsize=(16,8))
plt.title('Hydrogen Long-Short Term Memory (LSTM) Artificial Neural Network - PETR4')
plt.xlabel('Date', fontsize = 18)
plt.ylabel('Closing Price (BRL)', fontsize = 18)
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Historic (Train)','Validation', 'Predictions'], loc='lower right')
plt.show()



