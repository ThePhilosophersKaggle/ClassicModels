# -*- coding: utf-8 -*-
"""
Created on 12/03/2018

Wrapper module for NLP deep learning using Keras.

@author: The Philosophers
"""
from classic_models import *
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from gensim.models import KeyedVectors
from matplotlib import pyplot as plt
from keras.utils import plot_model
from matplotlib.image import imread
from sklearn.metrics import classification_report
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_absolute_error
import pickle


class NlpModels:
    def __init__(self, x_train, y_train, task, x_test=None, y_test=None, test_size=None, random_state=42, shuffle=True,
                 top_words=100000, max_len=200):
        """

        :param x_train: training text data (list, numpy array, pandas df values)
        :param y_train: label/corresponding value (list, numpy array, pandas df values)
        :param task: 'classification' or 'regression'.
        :param x_test: text test data.
        :param y_test: corresponding value of x_test test data.
        :param test_size: if no test set is given, one will be randomly selected from the training data.
        :param random_state: if test set is selected randomly among the training set, this parameter will
            fix a seed for reproducible results.
        :param shuffle: bool, if True training set will be shuffled before selecting the test set.
        :param top_words: how many words to keep in a dictionnary.
        :param max_len: maximum number of words as an input (longer samples will be truncated, smaller ones will be
            zero-padded).
       """
        # Check if either test set or test size was given
        incomplete_test_set = x_test is None or y_test is None
        if incomplete_test_set and test_size is None:
            raise Exception('You must either give a complete test set or ask that it be a proportion of the \
                            train set with the argument "test_size"')
        if incomplete_test_set:
            self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x_train, y_train,
                                                                                    test_size=test_size,
                                                                                    random_state=random_state,
                                                                                    shuffle=shuffle)
            # Get input as list (eases tokenizing and evaluation processes)
            self.x_test = list(self.x_test)
            self.x_train = list(self.x_train)
        else:
            self.x_train = x_train
            self.y_train = y_train
            self.x_test = x_test
            self.y_test = y_test
        # Memorize rest of values
        self.max_len = max_len
        self.top_words = top_words
        self.task = task
        # Format the labels according to the the task
        if task == "classification":
            self.y_train = to_categorical(self.y_train)
            self.y_test = to_categorical(self.y_test)
            # ToDo: Check if y_train and y_test have the same classes

    def nlp_cnn_lstm_pretrained_embeddings(self, loss, optimizer, batch_size,
                                                 nb_epochs, embeddings_source, path_to_embeddings, vector_dim,
                                                 save_model_as="cnn_lstm_pretrained", shuffle=True):
        """
        Builds a Convolutional Neural Network with a Long Short Term Memory recurrent layer. It uses pre-trained
            word embeddings to initialize the embedding matrix in the first layer, either from GloVe or FastText.  The
            best model will be saved then loaded before printing the results.

        x_train and x_test given at initialization will remain the same but inside this function they will
            be tokenized before fitting the model. The tokenizer will be saved in self.tokenizer.

        :param loss: loss function to be used. See Keras documentation for available functions, classical functions
            are: 'mse', 'categorical_crossentropy', 'binary_crossentropy'.
        :param optimizer: just use 'adam'. See keras documentation for available options.
        :param batch_size: number of samples to be processed at the same time.
        :param nb_epochs: number of passes through the data.
        :param embeddings_source: type of pre-trained vectors, available: 'glove', 'fasttext'.
        :param path_to_embeddings: path to the word embeddings. For fasttext it is a .vec file, for glove a .txt file.
        :param vector_dim: dimension of the pre-trained vectors chosen (e.g for fasttext trained on wikipedia it is 300)
        :param save_model_as: as the model will be automatically saved, this is the name that will be used.
        :param shuffle: if True the train data will be shuffled before training.
        :return: will update the object will the following attributes: model, history. See keras documentation.
        """
        # Format data
        print("Tokenizing data...")
        self.tokenizer, x_train, x_test = tokenize(self.x_train, self.top_words, self.max_len, test=self.x_test)

        # load glove embeddings
        if embeddings_source == "fasttext":
            embeddings = load_fasttext_embeddings(path=path_to_embeddings,
                                                  embedding_dim=vector_dim, word_index=self.tokenizer.word_index,
                                                  max_words=None)

        elif embeddings_source == "glove":
            embeddings = load_glove_embeddings(path=path_to_embeddings,
                                               embedding_dim=vector_dim, word_index=self.tokenizer.word_index)
        else:
            raise Exception("Embeddings source not understood. Available: 'fasttext', 'glove'.")

        # Build model
        model = Sequential()
        model.add(Embedding(input_dim=len(self.tokenizer.word_index) + 1, output_dim=vector_dim,
                            weights=[embeddings], input_length=self.max_len, trainable=True))
        model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
        model.add(MaxPooling1D(pool_size=2))
        model.add(LSTM(units=100, return_sequences=False))
        model.add(Dense(50))
        output_layer(model, self.task, self.y_train)
        model.compile(loss=loss, optimizer=optimizer, metrics=["acc"])
        print(model.summary())
        checkpointer = ModelCheckpoint(filepath=save_model_as + ".hdf5", verbose=1, monitor='val_acc',
                                       save_best_only=True)
        early_stop = EarlyStopping(monitor='val_acc', min_delta=0, patience=3, verbose=0, mode='auto')
        history = model.fit(x_train, self.y_train, epochs=nb_epochs, batch_size=batch_size,
                            callbacks=[checkpointer, early_stop],
                            shuffle=shuffle, validation_data=(x_test, self.y_test))
        self.model = model
        self.history = history
        try:
            model.load_weights(save_model_as + ".hdf5")
            self.model = model
        except:
            print("Best model couldn't be loaded.")

    def predict(self, text_data, show_proba=False):
        # Put text data into a list if it is not (else each letter is considered as an element of a list during
        # tokenizing)
        if type(text_data) != list:
            text_data = [text_data]
        # Pre-process the data
        sequences = self.tokenizer.texts_to_sequences(text_data)
        text_data = pad_sequences(sequences, maxlen=self.max_len)
        # Return prediction
        if self.task == "classification":
            if not show_proba:
                return np.argmax(self.model.predict(text_data), axis=1)
        return self.model.predict(text_data)

    def visualize_model_graph(self, graph_name="nlp_model.png"):
        """
        Plots saves the model graph as a png image and plots it.
        :param graph_name: path and name of the png image file that is going to be saved then plotted.
        :return: plots the image.
        """
        plot_model(self.model, to_file=graph_name)
        img = imread(graph_name)
        imgplot = plt.imshow(img)
        plt.show()

    def show_training_history(self):
        """
        plots a graph of training history accross epoch. Will plot val_accuracy and val_loss if available.
        :return: plots the graph.
        """
        print_keras_training_history(self.history)

    def evaluate(self, mapping_function=None, target_names=None):
        """
        Evaluates the model using classic metrics (classification report from scikit learn if classification problem).

        :param mapping_function: fucntion that will be applied for prediction. For example in a classification problem
            classes are 1 - 5, taking the argmax of a prediction will give 1-4, so you would want to apply
            mapping_function = lambda x: x + 1.
        :param target_names: if a classification task these will be the names of the classes.
        :return: prints the classification report.
        """
        if mapping_function is not None:
            y_pred = mapping_function(self.predict(self.x_test))
        else:
            y_pred = self.predict(self.x_test)
        if self.task == "classification":
            print(classification_report(np.argmax(self.y_test.values, axis=1), y_pred, target_names=target_names))
        elif self.task == "regression":
            print("Mean absolute error: ", mean_absolute_error(self.y_test, y_pred))
            print("explained variance score: ", explained_variance_score(self.y_test, y_pred))

    def test_on_new_data(self, x_test, y_test, mapping_function=None, target_names=None):
        """

        :param x_test: list of strings
        :param y_test: corresponding value (label...). Raw vector (e.g. [0, 1, 0, 2 ...]
        :param mapping_function: function to be applied on the prediction.
        :param target_names: if classification problem these will be the names of the classes.
        :return: prints classical metrics.
        """
        if mapping_function is not None:
            y_pred = mapping_function(self.predict(x_test))
        else:
            y_pred = self.predict(x_test)
        if self.task == "classification":
            print(classification_report(y_test, y_pred, target_names=target_names))
        elif self.task == "regression":
            print("Mean absolute error: ", mean_absolute_error(y_test, y_pred))
            print("explained variance score: ", explained_variance_score(y_test, y_pred))


def load_glove_embeddings(path, embedding_dim, word_index):
    """
    Loads GloVe embeddings for weight initialization in the embedding layer.

    :param path: path to glove embedding matrix.
    :param embedding_dim: dimension of the GloVe embedding matrix.
    :param word_index: the word index obtained using tokenizer.word_index
        (from keras.preprocessing.text import Tokenizer)
    :return: the embedding matrix that can be used to initialize the embedding layer for
        NLP in Keras. e.g.:
                    model.add(Embedding(input_dim=len(word_index) + 1, output_dim=dim, weights=[embeddings],
                    input_length=max_len, trainable=True))
    """
    embedding_index = {}
    f = open(path, encoding="utf8")
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embedding_index[word] = coefs
    f.close()
    print('Found %s word vectors.' % len(embedding_index))
    return create_embedding_matrix(word_index=word_index, embedding_index=embedding_index, embedding_dim=embedding_dim)


def create_embedding_matrix(word_index, embedding_index, embedding_dim):
    """
    Creates the embedding matrix that can be used to initialize the embedding layer in Keras (e.g. initialize it with
        pre-trained GloVe or FastText word vectors). The vectors must have been read first.
    :param word_index: word index from the Keras' tokenizer.
    :param embedding_index: index of the pre-trained word in the list read from its storing file.
    :param embedding_dim: dimension of the vectors.
    :return: embedding matrix
    """
    embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
    for word, i in word_index.items():
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    return embedding_matrix


def load_fasttext_embeddings(path, embedding_dim, word_index, max_words=None):
    """
    Loads GloVe embeddings for weight initialization in the embedding layer.

    :param path: path to glove embedding matrix.
    :param embedding_dim: dimension of the GloVe embedding matrix.
    :param word_index: the word index obtained using tokenizer.word_index
        (from keras.preprocessing.text import Tokenizer)
    :param max_words: maximum number of words to load (wiki data set is 10 GB).
    :return: the embedding matrix that can be used to initialize the embedding layer for
        NLP in Keras. e.g.:
                    model.add(Embedding(input_dim=len(word_index) + 1, output_dim=dim, weights=[embeddings],
                    input_length=max_len, trainable=True))
    """
    embedding_index = {}
    model = KeyedVectors.load_word2vec_format(path, limit=max_words)
    print("Reading fasttext vectors...")
    for word in model.vocab:
        embedding_index[word] = model[word]
    print('Found %s word vectors.' % len(embedding_index))
    return create_embedding_matrix(word_index=word_index, embedding_index=embedding_index, embedding_dim=embedding_dim)


def tokenize(train, top_words, max_len, test=None):
    """
    Preprocesses the text so it can then be fed to an embedding layer.

    :param train: list of strings.
    :param top_words:
    :param max_len: length of a sequence (i.e. len(train[i]).
    :param test: list of strings, default is None.
    :return: tokenizer, processed train data, processed test data
    """
    tokenizer = Tokenizer(num_words=top_words)
    tokenizer.fit_on_texts(train)
    train_sequences = tokenizer.texts_to_sequences(train)
    if test is not None:
        test_sequences = tokenizer.texts_to_sequences(test)
        return tokenizer, pad_sequences(train_sequences, maxlen=max_len), pad_sequences(test_sequences, maxlen=max_len)
    else:
        return tokenizer, pad_sequences(train_sequences, maxlen=max_len)
