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
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM

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

# Testing: 1
#df.plot(x='Date', y='Adj Close', kind='line')
#plt.show()

df.dropna(inplace=True) # Remove NaN lines from stock market holydays
#df.sort_values(by='Date', inplace=True, ascending=False) # Sorting for last recent data

################################## Variables Settup ########################################################

forecast_days = 7
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

################################ Data Preprocessing ######################################################


# NaN treatment
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(df[['Delta','Rsi']])
df[['Delta','Rsi']] = imputer.transform(df[['Delta','Rsi']])

# Column prediction using '-n' days for forecasting
df['Prediction'] = df[['Close']].shift(-forecast_days)