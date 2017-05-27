#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 19:52:17 2017

@author: Pranjal
"""

import numpy as np
import pandas as pd

data_train = pd.read_csv('train.csv')
data_features = pd.read_csv('features.csv')
data_stores = pd.read_csv('stores.csv')

data = pd.merge(data_stores, data_features,on='Store')
train_data = pd.merge(data, data_train, on=['Store','Date','IsHoliday'])
x = train_data.iloc[:, :-1].values
y = train_data.iloc[:, 10].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
encode_x = LabelEncoder();
x[:, 1] = encode_x.fit_transform(x[:, 1])
x[:, 8] = encode_x.fit_transform(x[:, 8])
x = x[:, (0,1,2,8,9)]

onehotencoder = OneHotEncoder(categorical_features = [0])
x = onehotencoder.fit_transform(x).toarray()
x = x[:, 1:]
onehotencoder = OneHotEncoder(categorical_features = [47])
x = onehotencoder.fit_transform(x).toarray()
x = x[:, 1:]

y = y.reshape(-1, 1)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
x = scaler.fit_transform(x)
y = scaler.fit_transform(y)

from sklearn.cross_validation import ShuffleSplit
bs = ShuffleSplit(344667, test_size = 0.1, random_state = 0)

for train_index, test_index in bs:
    print ("TRAIN:", train_index, "TEST:", test_index)

x_train = x[train_index]
y_train = y[train_index]
x_test = x[test_index]
y_test = y[test_index]
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
def net():
    model = Sequential()
    model.add(Dense(140, input_dim = 127,activation = 'relu',kernel_initializer = 'normal'))
    model.add(Dense(output_dim= 70,activation = 'relu',kernel_initializer = 'normal'))
    model.add(Dense(output_dim= 35,activation = 'relu',kernel_initializer = 'normal'))
    model.add(Dense(1,activation = 'linear',kernel_initializer = 'normal'))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

regressor = KerasRegressor(build_fn=net)
regressor.fit(x_train, y_train, epochs = 20, batch_size=10, validation_data = [x_test, y_test])

y_pred = regressor.predict(x_test)
print(r2_score(y_pred, y_test))




