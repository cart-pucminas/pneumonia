from __future__ import absolute_import, division, print_function

import pathlib
import pandas as pd
import numpy as np
from random import randint
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import math


def build_model(train_dataset):
      model = keras.Sequential([
            layers.Dense(1000, activation=tf.nn.relu, kernel_regularizer=keras.regularizers.l2(0.001), input_shape=[len(train_dataset.keys())]),
            keras.layers.Dropout(0.5),
            layers.Dense(100, activation=tf.nn.relu, kernel_regularizer=keras.regularizers.l2(0.001)),
            keras.layers.Dropout(0.5),
            layers.Dense(1)
            ])
      optimizer = tf.keras.optimizers.RMSprop(0.001)
      model.compile(loss='mean_squared_error',
                  optimizer=optimizer,
                  metrics=['mean_absolute_error', 'mean_squared_error'])
      model.summary()
      return model

def ANN(path, output, random):

      raw_dataset = pd.read_csv(filepath_or_buffer = path, sep=';', low_memory=False)
      del raw_dataset['CODE']

      dataset = raw_dataset.copy()
      origin = dataset.pop('RVC')
      dataset['RVC_yes'] = (origin == 'Yes')*1.0
      dataset['RVC_no'] = (origin == 'No')*1.0
      dataset.tail()

      NUM_WORDS = 10000
      (train_data, train_labels), (test_data, test_labels) = keras.datasets.imdb.load_data(num_words=NUM_WORDS)

      train_dataset = dataset.sample(frac=0.6667,random_state=random)
      test_dataset = dataset.drop(train_dataset.index)

      train_stats = train_dataset.describe()
      train_stats.pop(output)
      train_stats = train_stats.transpose()

      train_labels = train_dataset.pop(output)
      test_labels = test_dataset.pop(output)
      
      def norm(x):
            return ((x - train_stats['mean']) / train_stats['std'])

      normed_train_data = norm(train_dataset)
      normed_test_data = norm(test_dataset)

      model = build_model(train_dataset)
      early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=100)

      history = model.fit(normed_train_data, train_labels, epochs=2000,
                          validation_split = 0.2, verbose=0)
                          

      loss, mae, mse = model.evaluate(normed_test_data, test_labels, verbose=0)

      test_predictions = model.predict(normed_test_data).flatten()

      errors = abs(test_predictions - test_labels)
      mape = 100 * (errors / test_labels)
      accuracy = 100 - np.mean(mape)
      
      new_test_predictions = []
      for i in test_predictions:
            new_test_predictions.append(round(i))

      new_test_labels = []
      for j in test_labels:
            new_test_labels.append(round(j))

      result = pd.DataFrame()
      result['Actual Value'] = new_test_labels
      result['Prediction'] = new_test_predictions
      return result, accuracy


random = randint(0,1000)
result, accuracy = ANN('../Experiment 1 - Categorization from percentage ranges of the number of children dead by pneumonia/DATABASE_EXPERIMENT_1.csv', 'NDN', random)
result.to_csv('../Results/ANN_RESULTS_EXPERIMENT1.csv' , sep=';',  encoding='utf-8')
print("==============================================================================================================================================================")
print("Accuracy: " + str(accuracy))
print("The complete result will be saved in the folder: Results, with name: ANN_RESULTS_EXPERIMENT1.csv")


