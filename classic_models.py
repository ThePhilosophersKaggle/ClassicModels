# -*- coding: utf-8 -*-
"""
Created on 12/03/2018

Wrapper module for deep learning using Keras.

@author: The Philosophers
"""
import pandas as pd
from keras.layers import Dense
from matplotlib import pyplot as plt


def to_categorical(y_label):
    """
    Uses pandas' get dummies to have a sparse binary matrix which columns correspond to classes and each row has a
        a boolean value to say if that row is in the column's class or not.
    """
    return pd.get_dummies(y_label)


def output_layer(model, task, y_train):
    """
    Creates the output layer of a Keras deep learning model by choosing the dimension of the Dense layer depending on
        the task: if classification there will be as many neurons as there are classes, if regression there will be
        only one. Activation is "sigmoid" for classification, there is no activation for regression.
        Todo: add constraints over values for regression purposes.
    :param model: keras sequential model.
    :param task: type of task, available: 'regression', 'classification'.
    :param y_train: labels of training data.
    :return: adds the layer to the model (model.add)
    """
    if task == "classification":
        model.add(Dense(len(y_train.columns), activation='sigmoid'))
    elif task == "regression":
        model.add(Dense(1))
    else:
        raise Exception("Available tasks: 'regression', 'classification'. You entered: ", task)


def print_keras_training_history(history):
    plt.figure()
    keys = history.history.keys()  # list all data in history
    if "acc" in keys:
        plt.plot(history.history['acc'])
    if 'val_acc' in keys:
        plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    plt.figure()
    # summarize history for loss
    if "loss" in keys:
        plt.plot(history.history['loss'])
    if "val_loss" in keys:
        plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
