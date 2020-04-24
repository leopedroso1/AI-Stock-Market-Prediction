# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 14:59:01 2020

@author: Leonardo Pedroso dos Santos
"""

# Support libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Ploting - Testing
#import plotly.io as pio
#import plotly.graph_objs as go
#import plotly.offline as py
#from plotly.offline import plot, iplot

# Preprocessing libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer

# Machine Learning Models
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVR

# Data Analysis 
# TO DO >> Create a classification 'buy', 'hold', 'short' given some % margin
# TO DO >> Data set for fundamentalist analysis

#py.init_notebook_mode(connected = True)

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

## Feature data set - dependent variables (X) and convert it to a numpy array and remove the last 'n' rows/days to be predicted
X = df.drop(['Date', 'Prediction'], 1)[:-forecast_days]
#
## Target data set - independent variable (y) and convert it to a numpy array
y = np.array(df['Prediction'])[:-forecast_days]

## Split data 75% training 25% testing
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0) 

#print(df.dtypes)

################################ Machine Learning Models #################################################

# Predictions >>> Try to predict the forecast for the next 7 days 

#### Model 1 >> Regression >> Decision Tree Regression ###
decision_tree = DecisionTreeRegressor().fit(x_train, y_train)

#### Model 2 >> Regression >> Linear Regression
linear_regression = LinearRegression().fit(x_train, y_train)

### Model 3 >> Regression >> Random Forest Regressor ###
random_forest_reg = RandomForestRegressor(n_estimators= 100, random_state= 0).fit(x_train, y_train)

### Model 4 >> Regression >> Support Vector Regressor ###
svr_regressor = SVR(kernel= 'linear').fit(x_train, y_train) #Parameter kernel: 'linear', 'poly', 'sigmoid', 'rbf' (Gaussian) -  Default
# Note: Gaussian kernel exhibited better results

x_future = df.drop(['Date','Prediction'], 1)[:-forecast_days]
x_future = x_future.tail(forecast_days)
x_future = np.array(x_future)


## Tree Regressor Prediction >> 
tree_prediction = decision_tree.predict(x_future)

tree_predictions = tree_prediction
valid_tree_pred = df[X.shape[0]:]
valid_tree_pred['Predictions'] = tree_predictions


## Linear Regression Prediction >> 
lr_prediction = linear_regression.predict(x_future)

lr_predictions = lr_prediction
valid_lr_pred = df[X.shape[0]:]
valid_lr_pred['Predictions'] = lr_predictions


## Random Forest Regressor >>
random_forest_pred = random_forest_reg.predict(x_future)

rand_forest_predictions = random_forest_pred
valid_rf_pred = df[X.shape[0]:]
valid_rf_pred['Predictions'] = rand_forest_predictions

## Support Vector Regressor >> 
svr_pred = svr_regressor.predict(x_future)

svr_predictions = svr_pred
valid_svr_pred = df[X.shape[0]:]
valid_svr_pred['Predictions'] = svr_predictions

# Classifiers >> Try to establish a classification given our parameters
# UNDER CONSTRUCTION #