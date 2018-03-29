# -*- coding: utf-8 -*-
"""
Created on 12/03/2018
Wrapper module for time series deep learning using Keras.
@author: The Philosophers
"""
import numpy as np
import pandas as pd
from classic_models import *
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Reshape
from keras.layers import Dropout
from keras.layers import Merge
from keras.layers import Concatenate
from matplotlib import pyplot as plt
from keras.layers import Conv1D
from keras.models import Sequential, Model
from keras.layers import Concatenate, Dense, LSTM, Input, concatenate
from keras.layers import MaxPooling1D
import math
from sklearn.metrics import mean_squared_error
from keras import backend

class TimeSeriesModels:
    def __init__(self, pandas_dataframe, dates_column, target_column, regressors=None, train_test_split=0.66, seed=7,
                 look_back=1, look_forward=1, interval=0):
        """
        Initialises the Time Series object and takes care of the preprocessing.

        :param pandas_dataframe: pandas dataframe containing a column with the time series values, a column with the
            dates, and columns containing additional features/regressors if any.
        :param dates_column: name of the column containing the dates.
        :param target_column: column containing the values to predict.
        :param regressors: name of the columns to use as regressors (type: list)
        :param train_test_split: percentage of data to use for training.
        :param seed: seed for reproducible results.
        :param look_back: how many data points to use for prediction.
        :param look_forward: how many data points to predict.
        :param interval: interval for differentiating the data to make it stationary.
        """
        data = pd.DataFrame(index=pandas_dataframe[dates_column].values, data=pandas_dataframe[target_column].values)
        # Calculate the training set size
        train_size = int(len(data)*train_test_split)
        # Scale the data pre-train/test split
        scaler = MinMaxScaler(feature_range=(0, 1))
        self.scaler = scaler
        data = scaler.fit_transform(data)
        # Get the time series as stationary (for the given interval, if 0 don't make it a series of 0)
        if interval > 0:
            data = difference(data, interval)
        # Map the series to a supervised problem (values for days 1-n with regressors for these days to predict days
        # n + 1 ... n + k
        x, y = timeseries_to_supervised(data, look_back=look_back, look_forward=look_forward)
        # Split train and test
        self.x_train, self.y_train = x[:train_size], y[:train_size]
        self.x_test, self.y_test = x[train_size:], y[train_size:]
        # Use regressors if required
        if regressors is not None:
            self.x_train, self.x_test = add_regressors(self.x_train, self.x_test, regressors, pandas_dataframe,
                                                       dates_column, look_forward, look_back)
        # Set last attributes
        self.seed = seed
        self.look_back = look_back
        self.look_forward = look_forward
        self.regressors = regressors

    def lstm(self, neurons, nb_epochs, verbose=0):
        # scaler, train_scaled, test_scaled = scale(self.x_train, self.x_test)
        x_train = self.x_train.reshape(self.x_train.shape[0], 1, self.look_back + 1)
        model = Sequential()
        model.add(LSTM(neurons, batch_input_shape=(1, x_train.shape[1], x_train.shape[2]), stateful=True))
        model.add(Dense(self.look_forward))
        model.compile(loss='mean_squared_error', optimizer='adam')
        print(model.summary())
        history = []
        for i in range(nb_epochs):
            history.append(model.fit(x_train, self.y_train, epochs=1, batch_size=1, verbose=verbose, shuffle=False))
            model.reset_states()
            if i % int(nb_epochs/10) == 0:
                print("computation at: ", int(100*i/nb_epochs), "%")
        self.model = model

    def deep_lstm(self, neurons, nb_epochs, verbose=0):
        # scaler, train_scaled, test_scaled = scale(self.x_train, self.x_test)
        x_train = self.x_train.reshape(self.x_train.shape[0], 1, self.look_back + 1)
        model = Sequential()
        model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu',
                         batch_input_shape=(1, x_train.shape[1], x_train.shape[2])))
        # model.add(MaxPooling1D(pool_size=2))
        model.add(LSTM(neurons, stateful=True))
        model.add(Dense(50))
        model.add(Dense(50, activation="softmax"))
        model.add(Dense(self.look_forward, activation='relu'))
        model.compile(loss='mean_squared_error', optimizer='adam')
        print(model.summary())
        history = []
        for i in range(nb_epochs):
            history.append(model.fit(x_train, self.y_train, epochs=1, batch_size=1, verbose=verbose, shuffle=False))
            model.reset_states()
            if i % int(nb_epochs / 10) == 0:
                print("computation at: ", int(100 * i / nb_epochs), "%")
        self.model = model

    def lstm_reg(self, neurons, nb_epochs, verbose=0):
        # scaler, train_scaled, test_scaled = scale(self.x_train, self.x_test)
        x_train = self.x_train.reshape(self.x_train.shape[0], len(self.regressors) + 1, self.look_back + 1)
        model = Sequential()
        model.add(LSTM(neurons, batch_input_shape=(1, x_train.shape[1], x_train.shape[2]), stateful=True))
        model.add(Dense(self.look_forward))
        model.compile(loss='mean_squared_error', optimizer='adam')
        print(model.summary())
        # history = []
        for i in range(nb_epochs):
            model.fit(x_train, self.y_train, epochs=1, batch_size=1, verbose=verbose, shuffle=False)
            model.reset_states()
            if i % int(nb_epochs/10) == 0:
                print("computation at: ", int(100*i/nb_epochs), "%")
        self.model = model

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

    def lstm_reg_test(self, neurons, nb_epochs, verbose=0, dropout=0, recurrent_dropout=0):
        # scaler, train_scaled, test_scaled = scale(self.x_train, self.x_test)
        x_train = self.x_train.reshape(self.x_train.shape[0], len(self.regressors) + 1, self.look_back + 1)
        x_train_autoreg = np.reshape(x_train[:, :, 0], (x_train.shape[0], x_train.shape[1], 1))
        print(x_train_autoreg.shape)
        # Evaluate time series
        time = Input(batch_shape=(1, x_train.shape[1], 1))
        time_lstm = LSTM(neurons, dropout=dropout, recurrent_dropout=recurrent_dropout, stateful=True,
                         return_sequences=True)(time)
        time_lstm = LSTM(neurons, dropout=dropout, recurrent_dropout=recurrent_dropout, stateful=True,
                         return_sequences=True)(time_lstm)
        time_lstm = LSTM(neurons, dropout=dropout, recurrent_dropout=recurrent_dropout, stateful=True,
                         return_sequences=False)(time_lstm)

        # Take regressors into account
        regressor_processing = Input(batch_shape=(1, x_train.shape[1], x_train.shape[2] - 1))
        regressor_processing_layers = Reshape((x_train.shape[1]*(x_train.shape[2] - 1),))(regressor_processing)
        regressor_processing_layers = Dense((x_train.shape[1]*(x_train.shape[2] - 1)),
                                            activation="softmax")(regressor_processing_layers)
        regressor_processing_layers = Dropout(0.2)(regressor_processing_layers)
        regressor_processing_layers = Dense(neurons, activation="sigmoid")(regressor_processing_layers)

        # merge
        merge = concatenate([time_lstm, regressor_processing_layers])
        merge = Dense(50, activation="relu")(merge)
        merge = Dropout(0.2)(merge)
        merge = Dense(25, activation="sigmoid")(merge)
        merge = Dropout(0.2)(merge)
        merge = Dense(10, activation="relu")(merge)
        merge = Dropout(0.2)(merge)
        merge = Dense(1, activation="sigmoid")(merge)
        model = Model(inputs=[time, regressor_processing], outputs=merge)
        model.compile(optimizer='adam', loss='mean_squared_error',
                      metrics=['accuracy'])

        print(model.summary())
        history = []
        for i in range(nb_epochs):
            history.append(model.fit([x_train_autoreg, x_train[:, :, 1:]], self.y_train, epochs=1, batch_size=1,
                                     verbose=verbose, shuffle=False))
            model.reset_states()
            if i % int(nb_epochs / 10) == 0:
                print("computation at: ", int(100 * i / nb_epochs), "%")
                print(history[i].history)
        self.history = history
        self.model = model

    def plot_train_data(self):
        plt.plot(self.x_train[:, 0, 0])
        plt.show()

    def show_training_history(self):
        """
        plots a graph of training history accross epoch. Will plot val_accuracy and val_loss if available.
        :return: plots the graph.
        """
        hist = [i.history["loss"][0] for i in self.history]
        plt.plot(hist)


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


def scale(train, test):
    """
    rescale the data (usually between 0 and 1) for better performances.
    :param train: training data
    :param test: test data
    :return: scaler, training data scaled, test data scaled
    """
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
    """
    Invert operation from the scaling operation. Apparently we are not using this at all.

    :param scaler: obtained with the "scale" function.
    :param X: data to rescale.
    :param value: no idea
    :return: rescaled data
    """
    new_row = [x for x in X] + [value]
    array = np.array(new_row)
    array = array.reshape(1, len(array))
    inverted = scaler.inverse_transform(array)
    return inverted[0, -1]


def difference(data, interval=1):
    """
    Makes the data stationary by differentiating t+interval - t.

    :param data: data as numpy array (or anything that can be fed to a Pandas dataframe)
    :param interval: integer
    :return: differentiated data
    """
    df = pd.DataFrame(data)
    return df.shift(interval).values - df.values


def add_regressors(x_train, x_test, regressors, pandas_dataframe, dates_column, look_forward, look_back):
    """
    Reshape data to add a 3rd dimension for regressors (=> order 2 tensor)

    :param x_train: training data
    :param x_test: test data
    :param regressors: list of names of regressors to use
    :param pandas_dataframe: initial pandas dataframe containing the regressors
    :param dates_column: column containing the dates linked to the target variable
    :param look_forward: how many data points to use for prediction
    :param look_back: how many data points to predict
    :return:
    """
    nb_regressors = len(regressors)
    train_size = x_train.shape[0]
    x_train = x_train.reshape([x_train.shape[0], x_train.shape[1], 1])
    x_test = x_test.reshape([x_test.shape[0], x_test.shape[1], 1])
    # Initialize the regressor layers to be stacked on to the 3rd dimension of the data matrix
    reg_x_train = np.zeros([x_train.shape[0], x_train.shape[1], nb_regressors])
    reg_x_test = np.zeros([x_test.shape[0], x_test.shape[1], nb_regressors])
    # x_train dans x_test are matrixes resulting from the mapping to a supervised learning problem.
    # As the regressors are still a series, they have to be reshaped to match the supervised problem created
    # above
    for i in range(nb_regressors):
        reg_x_tmp, reg_y_tmp = timeseries_to_supervised(
            pd.DataFrame(index=pandas_dataframe[dates_column].values,
                         data=pandas_dataframe[regressors[i]].values),
            look_back=look_back, look_forward=look_forward)
        # Split train and test
        reg_x_train[:, :, i] = reg_x_tmp[:train_size]
        reg_x_test[:, :, i] = reg_x_tmp[train_size:]
    # Append the regressors to the data shaped for supervised learning
    x_train = np.append(x_train, reg_x_train, axis=2)
    x_test = np.append(x_test, reg_x_test, axis=2)
    return x_train, x_test
