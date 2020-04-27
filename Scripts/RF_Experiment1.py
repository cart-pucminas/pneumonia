from __future__ import absolute_import, division, print_function

import pathlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from random import randint
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
import math
from sklearn.ensemble import RandomForestRegressor

def RF(caminho, coluna, random):

      raw_dataset = pd.read_csv(filepath_or_buffer = caminho, sep=';', low_memory=False)
      del raw_dataset['CODE']

      dataset = raw_dataset.copy()
      origin = dataset.pop('RVC')
      dataset['yes'] = (origin == 'Yes')*1.0
      dataset['no'] = (origin == 'No')*1.0
      dataset.tail()

      features = np.array(dataset)
      NUM_WORDS = 10000
      (train_data, train_labels), (test_data, test_labels) = keras.datasets.imdb.load_data(num_words=NUM_WORDS)
      train_dataset = dataset.sample(frac=0.6667,random_state=random)
      test_dataset = dataset.drop(train_dataset.index)

      train_stats = train_dataset.describe()
      train_stats.pop(coluna)
      train_stats = train_stats.transpose()
      train_labels = train_dataset.pop(coluna)
      test_labels = test_dataset.pop(coluna)
      
      def norm(x):
            return ((x - train_stats['mean']) / train_stats['std'])
      normed_train_data = norm(train_dataset)
      normed_test_data = norm(test_dataset)
      print(normed_test_data)
      test_features = normed_test_data
      train_features = normed_train_data
      rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
      
      rf.fit(train_features, train_labels)
      predictions = rf.predict(test_features)
      errors = abs(predictions - test_labels)
      
      mape = 100 * (errors / test_labels)
      accuracy = 100 - np.mean(mape)

      new_test_predictions = []
      for i in predictions:
            new_test_predictions.append(round(i))

      results = pd.DataFrame()
      results['ACTUAL VALUE'] = test_labels
      results['PREDICTION'] = new_test_predictions
      return results, accuracy



random = randint(0,1000)
result, accuracy = RF('../Experiment 1 - Categorization from percentage ranges of the number of children dead by pneumonia/DATABASE_EXPERIMENT_1.csv', 'NDN', random)
result.to_csv('../Results/RF_RESULTS_EXPERIMENT1.csv' , sep=';',  encoding='utf-8')
print("==============================================================================================================================================================")
print("Accuracy: " + str(accuracy) + "%")
print("The complete result will be saved in the folder: Results, with name: RF_RESULTS_EXPERIMENT1.csv")
