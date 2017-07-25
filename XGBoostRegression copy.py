#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 23 14:27:08 2017

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

from sklearn.cross_validation import ShuffleSplit
bs = ShuffleSplit(344667, test_size = 0.1, random_state = 0)

for train_index, test_index in bs:
    print ("TRAIN:", train_index, "TEST:", test_index)

x_train = x[train_index]
y_train = y[train_index]
x_test = x[test_index]
y_test = y[test_index]

from xgboost import XGBRegressor
from sklearn.metrics import r2_score
import time
start = time.time()
regressor = XGBRegressor(max_depth=20, n_estimators=600, n_jobs=-1, silent=0)
regressor.fit(x_train, y_train)
elapsed = time.time() - start

y_pred = regressor.predict(x_test)
r2_score(y_pred, y_test)