# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 10:57:27 2020

@author: Leonardo
"""

import OpenBlender # News and other datasets
import pandas as pd
import json


# Goals Step by Step
#Step 1: Prepare set up: Scikit Learn + Pandas + Open Blender
#Step 2: Get the data for daily Apple Stock since 2017
#Step 3: Define and understand target for ML
#Step 4: Blend business news to our data
#Step 5: Prepare our data and apply ML
#Step 6: Measure and analyze results
#Step 7: Break the data and train/test through time


# Main Goal: detect if the price is going to increase or decrease on the next day so we can buy or short.
# The ‘change’ is the percentual increase or decrease that happened between the opening and closing price, so it works for us.
# 1º attempt: + and - 0.5% variation on change

# Retrieve dataset from Openbender.io >> News from USA Today and The WSJ Journal
action = 'API_createTextVectorizerPlus'
parameters = {
    'token' : 'INSERT YOUR TOKEN HERE',
    'name' : 'Wall Street and USA Today Vectorizer',
    'sources':[
              {'id_dataset':"5e2ef74e9516294390e810a9", 
               'features' : ["text"]},
              {'id_dataset' : "5e32fd289516291e346c1726", 
               'features' : ["text"]}
    ],
    'ngram_range' : {'min' : 1, 'max' : 2},
    'language' : 'en',
    'remove_stop_words' : 'on',
    'min_count_limit' : 2
}
response = OpenBlender.call(action, parameters)
response

# Retrieve dataset from Openbender.io >> Apple Stock
action = 'API_getObservationsFromDataset'
interval = 60 * 60 * 24 # One day
parameters = { 
      'token':'INSERT YOUR TOKEN HERE', # Go to Openbender.IO > Sign in and retrieve your API key at 'Account' link
      'id_dataset':'5d4c39d09516290b01c8307b', # AAPL34
      'date_filter':{"start_date":"2017-01-01T06:00:00.000Z", # Start 
                     "end_date":"2020-02-09T06:00:00.000Z"}, # End date
      'aggregate_in_time_interval' : {
              'time_interval_size' : interval, 
              'output' : 'avg', 
              'empty_intervals' : 'impute'
      },
      'blends' :
       [{"id_blend" : "5e46c8cf9516297ce1ada712", # here is the blend from the previous News data frame
         "blend_class" : "closest_observation", 
         "restriction":"None",
         "blend_type":"text_ts",
         "specifications":{"time_interval_size" : interval}
       }],
       'lag_feature' : {'feature' : 'change', 'periods' : [-1]}
}
df = pd.read_json(json.dumps(OpenBlender.call(action, parameters)['sample']), convert_dates=False, convert_axes=False).sort_values('timestamp', ascending=False)
df.reset_index(drop=True, inplace=True)

print(df.shape)
df.head()


# Set parameters for sell, short or hold
# Where ‘change’ decreased more than 0.5%
df['negative_poc'] = [1 if val < 0.5 else 0 for val in df['lag-1_change']]

# Where ‘change’ increased more than 0.5%
df['positive_poc'] = [1 if val > 0.5 else 0 for val in df['lag-1_change']]
df[['lag-1_change', 'positive_poc', 'negative_poc']].head()

# ‘lag-1_change’: This feature aligns the ‘change’ values with the “previous day data”. Today x Yesterday stock variarion
#  The most recent observations is NaN because that’s what we would want to predict for ‘tomorrow’.

# ML libraries and performance checks
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn import metrics

# Define target and remove unwanted features to avoid data leakage
target = 'positive_poc'

df_positive = df.select_dtypes(['number'])

for rem in ['lag-1_change', 'negative_poc']:
    df_positive = df_positive.loc[:, df_positive.columns != rem]

# Create train/test sets
X = df_positive.loc[:, df_positive.columns != target].values # Dependent Values 
y = df_positive.loc[:,[target]].values # Independent Values

# Take first bit of the data as test and the last as train. The dataset is ordered by timestamp descending and we want to train with previous observations and test with subsequent ones.
div = int(round(len(X) * 0.29))

X_test = X[:div]
y_test = y[:div]

X_train = X[div:]
y_train = y[div:]

print('Train:')
print(X_train.shape)
print(y_train.shape)
print('Test:')
print(X_test.shape)
print(y_test.shape)


rf = RandomForestRegressor(n_estimators = 1000, random_state = 1)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)


print("AUC score:")
print(roc_auc_score(y_test, y_pred))
print('---')

# Make it binary and look at the confusion matrix + accuracy check
preds = [1 if val > 0.5 else 0 for val in y_pred]
print('Confusion Matrix:')
print(metrics.confusion_matrix(y_test, preds))
print('---')

print('Acurracy:')
print(accuracy_score(y_test, preds))
print('---')

# Accuracy Score: 0.78684
# Confusion Matrix:
# 157 24
# 67  62
# Accuracy: 0.706451

# Methodology: Retraing constantly our classifier given the market volatility and check if the results are consistent
results = []
for i in range(0, 90, 5): 

    time_chunk = i/100

     print(“time_chunk:” + str(time_chunk) + “ starts”)

     df_ml = df_positive[:int(round(df_positive.shape[0] * (time_chunk + 0.4)))]

     X = df_ml.loc[:, df_ml.columns != target].values
     y = df_ml.loc[:,[target]].values
 
    div = int(round(len(X) * 0.29))

     X_test = X[:div]
     y_test = y[:div]
     X_train = X[div:]
     y_train = y[div:]

     rf = RandomForestRegressor(n_estimators = 1000, random_state = 1)
     rf.fit(X_train, y_train)

     y_pred = rf.predict(X_test)

     preds = [1 if val > threshold else 0 for val in y_pred]

     try:
 
         roc = roc_auc_score(y_test, y_pred)
 
    except:

        roc = 0
 
    conf_mat = metrics.confusion_matrix(y_test, preds)
 
    accuracy = accuracy_score(y_test, preds)
 
    results.append({
            'roc' : roc,
            'accuracy' : accuracy,
            'conf_mat' : conf_mat,
            'time_chunk' : time_chunk
            })
           
           
results_df = pd.DataFrame(results)
results_df

## After 16 executions for Retraining our Random forest
##  Accuracy / confusion matrix / roc / time_chunk
# 0 0.701613 / [[50, 16], [21, 37]] / 0.733020 / 0.00
#16 0.706452 / [[157, 24], [67, 62]] / 0.786843 / 0.00