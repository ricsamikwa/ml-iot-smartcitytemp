import os
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=gpu,floatX=float32"

from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
import pandas
import math
import io
import keras
import requests
from matplotlib import pyplot
import matplotlib.patches as mpatches
from numpy import concatenate
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from matplotlib.lines import Line2D
from keras import metrics
from keras.layers import LeakyReLU
from numpy import save
from numpy import load
import utils
import csv


dataset = pandas.read_pickle("dataset/dataset.pkl")

scaler  = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

num_train_points = 1398 # approx 70% of the dataset
num_val_points = 300
start_row = num_train_points + num_val_points
start_row = 0

filename = 'naive_baseline3.csv'


for n in range(1,137):

    num_observations = n

    print("Number of observations: " + str(num_observations))


    final_temp, tran_dataset, lasttemp = utils.create_naive_test_data(dataset,num_observations,start_row)

    iterations = 100
    batch_size = 100

    # break
    print_flag = 0


    for t in range(iterations):
      # predictions
      start, end = utils.get_rolling_window_bounds(0, len(tran_dataset), batch_size, 16, t)
      testX= np.array(tran_dataset[start:end])
      testY = np.array(final_temp[start:end])
      testYL = np.array(lasttemp[start:end])

      unseen_X = testX.reshape((testX.shape[0], num_observations*2))

      testYL = testYL.reshape((len(testYL), 1))
      inv_predictions = concatenate((unseen_X[:,-1:],testYL), axis=1)

      inv_predictions = scaler.inverse_transform(inv_predictions)

      inv_predictions = inv_predictions[:,1]


      # invert scaling for actual temp levels
      testY = testY.reshape((len(testY), 1))
      inv_temp = concatenate((unseen_X[:, -1:],testY), axis=1)
    
      inv_temp = scaler.inverse_transform(inv_temp)
      inv_temp = inv_temp[:,1]
      
      
      # root mean squared error (RMSE)
      rmsee = sqrt(mean_squared_error(inv_temp, inv_predictions))
      # test_acc = 1 - (rmse/max(inv_temp))*100
      
      # Mean absolute error
      mae = keras.metrics.mean_absolute_error(inv_temp, inv_predictions)

      rmse = rmsee 
      
      msee = rmsee
      
      R = utils.R(inv_temp, inv_predictions)

      if print_flag == 0:
        print_flag =2
        print('MAE: ',mae.numpy())
        print(inv_temp[0], inv_predictions[0])
        print('R squared: ', R.numpy()) 
        print('Test RMSE: %.3f  ' % (rmse))
      # print('Test RMSE: %.3f %% ' % ((rmse/max(inv_temp))*100))

      with open('results/'+filename,'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([num_observations, rmse, rmsee,msee])
