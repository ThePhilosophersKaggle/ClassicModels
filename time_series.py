# -*- coding: utf-8 -*-
"""
Created on 12/03/2018

Wrapper module for time series deep learning using Keras.

@author: The Philosophers
"""
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler
import math
from sklearn.metrics import mean_squared_error
from keras import backend

class TimeSeriesModels:
    def __init__(self, pandas_dataframe, dates_column, target_column, regressors=None, train_test_split=0.66, seed=7, look_back=1,
                 look_forward=1):
        data = pd.DataFrame(index=pandas_dataframe[dates_column].values, data=pandas_dataframe[target_column].values)
        # Calculate the training set size
        train_size = int(len(data)*train_test_split)
        # Scale the data pre-train/test split
        scaler = MinMaxScaler(feature_range=(0, 1))
        self.scaler = scaler
        data = scaler.fit_transform(data)
        # Put the problem as a supervise problem (matrix with t1, t2, t3 ; t2, t3, t4 ; ...)
        x, y = timeseries_to_supervised(data, look_back=look_back, look_forward=look_forward)
        # Split train and test
        self.x_train, self.y_train = x[:train_size], y[:train_size]
        self.x_test, self.y_test = x[train_size:], y[train_size:]
        # Check Regressors exist
        if regressors is not None:
            self.x_train = self.x_train.reshape([self.x_train.shape[0], self.x_train.shape[1], 1])
            self.x_test = self.x_test.reshape([self.x_test.shape[0], self.x_test.shape[1], 1])
            # self.y_train = self.y_train.reshape([self.y_train.shape[0], self.y_train.shape[1], 1])
            # self.y_test = self.y_test.reshape([self.y_test.shape[0], self.y_test.shape[1], 1])
            reg_x_train = np.zeros([self.x_train.shape[0], self.x_train.shape[1], len(regressors)])
            reg_x_test = np.zeros([self.x_test.shape[0], self.x_test.shape[1], len(regressors)])
            # reg_y_train = np.zeros([self.y_train.shape[0], self.y_train.shape[1], len(regressors)])
            # reg_y_test = np.zeros([self.y_test.shape[0], self.y_test.shape[1], len(regressors)])
            for i in range(len(regressors)):
                reg_x_tmp, reg_y_tmp = timeseries_to_supervised(
                    pd.DataFrame(index=pandas_dataframe[dates_column].values,
                                 data=pandas_dataframe[regressors[i]].values),
                    look_back=look_back, look_forward=look_forward)
                # Split train and test
                reg_x_train[:, :, i] = reg_x_tmp[:train_size]
                reg_x_test[:, :, i] = reg_x_tmp[train_size:]
            self.x_train = np.append(self.x_train, reg_x_train, axis=2)
            self.x_test = np.append(self.x_test, reg_x_test, axis=2)
            # self.y_train = np.append(self.y_train, reg_y_train, axis=2)
            # self.y_test = np.append(self.y_test, reg_y_test, axis=2)
        # Set last attributes
        self.seed = seed
        self.look_back = look_back
        self.look_forward = look_forward
        self.regressors = regressors

    def lstm(self, neurons, nb_epochs, verbose=0):
        #scaler, train_scaled, test_scaled = scale(self.x_train, self.x_test)
        x_train = self.x_train.reshape(self.x_train.shape[0], 1, self.look_back + 1)
        model = Sequential()
        model.add(LSTM(neurons, batch_input_shape=(1, x_train.shape[1], x_train.shape[2]), stateful=True))
        model.add(Dense(self.look_forward))
        model.compile(loss='mean_squared_error', optimizer='adam')
        print(model.summary())
        #history = []
        for i in range(nb_epochs):
            model.fit(x_train, self.y_train, epochs=1, batch_size=1, verbose=verbose, shuffle=False)
            model.reset_states()
            if i % int(nb_epochs/10) == 0:
                print("computation at: ", int(100*i/nb_epochs), "%")
        self.model = model
        #self.history = history
        #self.scaler = scaler
        #self.x_train_scaled = train_scaled
        #self.x_test_scaled = test_scaled

    def lstm_reg(self, neurons, nb_epochs, verbose=0):
        #scaler, train_scaled, test_scaled = scale(self.x_train, self.x_test)
        x_train = self.x_train.reshape(self.x_train.shape[0], len(self.regressors) + 1, self.look_back + 1)
        model = Sequential()
        model.add(LSTM(neurons, batch_input_shape=(1, x_train.shape[1], x_train.shape[2]), stateful=True))
        model.add(Dense(self.look_forward))
        model.compile(loss='mean_squared_error', optimizer='adam')
        print(model.summary())
        #history = []
        for i in range(nb_epochs):
            model.fit(x_train, self.y_train, epochs=1, batch_size=1, verbose=verbose, shuffle=False)
            model.reset_states()
            if i % int(nb_epochs/10) == 0:
                print("computation at: ", int(100*i/nb_epochs), "%")
        self.model = model
        #self.history = history
        #self.scaler = scaler
        #self.x_train_scaled = train_scaled
        #self.x_test_scaled = test_scaled

    def evaluate(self):
        # Predict training
        trainPredict = self.model.predict(
            self.x_train.reshape(self.x_train.shape[0], 1, self.look_back + 1), batch_size=1)
        # Predict test
        testPredict = self.model.predict(self.x_test.reshape(self.x_test.shape[0], 1, self.look_back + 1),
                                         batch_size=1)
        # Invert scaling
        yhat_train = self.scaler.inverse_transform(trainPredict)
        yhat_test = self.scaler.inverse_transform(testPredict)
        return yhat_train, yhat_test

def create_dataset(dataset, look_back=1, look_forward=1):
    """

    convert an array of values into a dataset matrix
    :param dataset: time series as a pandas series.
    :param look_back: how many data points to use to predict the next ones.
    :param look_forward: how many data points to predict.
    :return: x, y : x is the data points to use to predict, y is the true corresponding ediction.
    """
    x, y = [], []
    for i in range(len(dataset)-look_back - look_forward):
        a = dataset[i: i+look_back]
        x.append(a)
        y.append(dataset[i + look_back: i + look_back + look_forward - 1])
    print(x)
    return np.array(x), np.array(y)


# frame a sequence as a supervised learning problem
def timeseries_to_supervised(data, look_back=1, look_forward=1):
    """

    :param data:
    :param look_back:
    :param look_forward:
    :return: x: numpy array, y:numpy array
    """
    df = pd.DataFrame(data)
    # x
    columns = [df.shift(-i) for i in range(0, look_back+1)]
    df_x = pd.concat(columns, axis=1)[:-look_back - look_forward - 1]
    # y
    columns = [df.shift(-i) for i in range(0, look_forward)]
    df_y = pd.concat(columns, axis=1)[look_back + 1:-look_forward]
    return df_x.values, df_y.values


# scale train and test data to [-1, 1]
def scale(train, test):
    # fit scaler
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = scaler.fit(train)
    # transform train
    train = train.reshape(train.shape[0], train.shape[1])
    train_scaled = scaler.transform(train)
    # transform test
    test = test.reshape(test.shape[0], test.shape[1])
    test_scaled = scaler.transform(test)
    return scaler, train_scaled, test_scaled


# inverse scaling for a forecasted value
def invert_scale(scaler, X, value):
    new_row = [x for x in X] + [value]
    array = np.array(new_row)
    array = array.reshape(1, len(array))
    inverted = scaler.inverse_transform(array)
    return inverted[0, -1]
