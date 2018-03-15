# -*- coding: utf-8 -*-
"""
Created on 12/03/2018

Wrapper module for time series deep learning using Keras.

@author: The Philosophers
"""
import numpy as np
import pandas as pd


class TimeSeriesModels:
    def __init__(self, pandas_dataframe, dates_column, target_column):
        self.data = pd.DataFrame(index=pandas_dataframe[dates_column], data=pandas_dataframe[dates_column])


def create_dataset(dataset, look_back=1, look_forward=1):
    # convert an array of values into a dataset matrix
    x, y = [], []
    for i in range(len(dataset)-look_back - look_forward):
        a = dataset.iloc[i: i+look_back]
        x.append(a)
        y.append(dataset.iloc[i + look_back: i + look_back + look_forward - 1])
    return np.array(x), np.array(y)

