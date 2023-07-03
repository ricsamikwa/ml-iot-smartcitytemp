# -*- coding: utf-8 -*-
"""LSTM_adaptive_ventilation.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1ILhnjZifseE76A6rfsyaSwCWHtLRHgfx

# Machine Learning-based Energy Optimisation in Smart City Temperature Sensors
"""

# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline

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
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Dropout
from tensorflow.python.keras.layers import Activation
from tensorflow.python.keras.layers import LSTM
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from matplotlib.lines import Line2D
from keras import metrics
from keras.layers import LeakyReLU

"""# Data pre-processing"""

# fix random seed for reproducibility
np.random.seed(10)

"""###Temp data"""

# github
df_temp = pandas.read_csv("https://raw.githubusercontent.com/ricsamikwa/short-term-flood-water-level-prediction/master/temp_prediction/temperature.csv")

#display first
df_temp.head()

#dataframe shape
df_temp.shape

"""### Humidity data"""

#get humidity data
df_humidity = pandas.read_csv("https://raw.githubusercontent.com/ricsamikwa/short-term-flood-water-level-prediction/master/temp_prediction/humidity.csv")

df_temp.shape

df_temp = df_temp.drop(columns=['time'])
# df_temp.shape

df_humidity = df_humidity.drop(columns=['time'])
df_humidity.shape

series_temp = []
# series_final_temp = []
for i in range(0,len(df_temp)):
    row = df_temp.iloc[i]

    for property in ["temperature_%03d" % i for i in range(0,46*3)]:
        series_temp.append(row[property])

series_humidity = []
for i in range(0,len(df_temp)):
    row = df_humidity.iloc[i]

    for property in ["humidity_%03d" % i for i in range(0,46*3)]:
        series_humidity.append(row[property])

new_df_temp = pandas.DataFrame(series_temp, columns=['temperature'])
new_df_temp.shape
# print(new_df_temp)

new_df_humidity = pandas.DataFrame(series_humidity, columns=['humidity'])
# new_df_humidity.shape

"""### Combining dataset"""

#dataset merged on date_time
dataset = pandas.merge(new_df_humidity, new_df_temp, left_index=True, right_index=True)

dataset.head()
dataset.shape

"""### Scaling dataset"""

# normalize the dataset (LSTMs are sensitive to the scale of the input data)
scaler  = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

"""# Training"""

plt.plot(dataset[0:137,1], color='#017C8F')

"""#### Dataset sliding window (Multivariate)"""

#Antonio
temp_list = dataset[:,0]

reshaped_temp_list = np.array(temp_list).reshape(-1, 138)
print(reshaped_temp_list.shape)
# print(reshaped_dataset)

exploded = []
for i in range(1,len(reshaped_temp_list)):
  input_array = reshaped_temp_list[0]
  row = np.array([(input_array[:i], input_array[i:]) for i in range(1, 138)])
  exploded.append(row)

exploded = np.array(exploded)
# print(exploded.shape)
# print(exploded[0][0])

final_temp = []
tran_dataset = []
def create_training_data(dataset,num_observations):
  rows = math.floor(len(dataset)/138)
  for i in range(rows):
    tran_dataset.append(dataset[138*i:(138*i)+num_observations,:])
    final_temp.append(dataset[(138*(i+1))-1,1])

#number of values to be observed when making a each forecast
num_observations = 137

# frame data as supervised learning
create_training_data(dataset, num_observations)

"""#### Spliting data to training and test set"""

num_train_points = 1398 # approx 70% of the dataset
num_val_points = 300

#separated datasets
trainX = np.array(tran_dataset[:num_train_points])
validateX= np.array(tran_dataset[num_train_points:num_train_points+num_val_points])
testX= np.array(tran_dataset[num_train_points+num_val_points:])
trainY = np.array(final_temp[:num_train_points])
validateY = np.array(final_temp[num_train_points:num_train_points+num_val_points])
testY = np.array(final_temp[num_train_points+num_val_points:])

print(trainX.shape, validateX.shape, testX.shape, len(trainY), len(validateY), len(testY))

"""#### Reshaping to LSTM input format"""

#forecast points check
# print(validateX[0])
# print(validateY[0])

"""## Model (LSTM)"""

#Setup the LSTM model (with multivariate input)

model = Sequential()
# model.add(LSTM(32, activation='relu', input_shape=(trainX.shape[1], trainX.shape[2])))
X = keras.layers.Input(shape=(None, trainX.shape[2])) # to be added in the sequential model
model.add(LSTM(32, activation='relu', input_shape=(None, trainX.shape[2])))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')

model.summary()

"""## Model Learning

### training and validation
"""

callback = EarlyStopping(monitor='val_loss', mode='min',verbose=1,patience=80, baseline=0.000073750)

#training :  epochs and batch size 64
history = model.fit(trainX, trainY, epochs=50, batch_size=64, callbacks=[callback],
                    validation_data=(validateX, validateY), shuffle=False)

"""###training performance"""

# plot loss during training
plt.figure(
    figsize=(7, 5))
pyplot.plot(history.history['loss'], label='training')
pyplot.plot(history.history['val_loss'], label='validation')
pyplot.title('Model train vs Validation loss')
plt.xlabel('no of epochs')
plt.ylabel('loss')
pyplot.legend(['Train', 'Validation'], loc='upper right')
pyplot.show()

# model RMSE
train_score = model.evaluate(trainX, trainY, verbose=1)
train_score = math.sqrt(train_score)

validation_score = model.evaluate(validateX, validateY, verbose=1)
validation_score = math.sqrt(validation_score)
print('Train Score: %.2f  RMSE' % (train_score))
print('Validation Score: %.2f  RMSE' % (validation_score))

"""# Testing

### predictions using unseen test data

# No padding
"""

num_observations = 70
final_temp = []
tran_dataset = []

num_train_points = 1398 # approx 70% of the dataset
num_val_points = 300

create_training_data(dataset, num_observations)

testX= np.array(tran_dataset[num_train_points+num_val_points:])
testY = np.array(final_temp[num_train_points+num_val_points:])

# predictions
predictions = model.predict(testX)
unseen_X = testX.reshape((testX.shape[0], num_observations*2))

inv_predictions = concatenate((unseen_X[:,-1:],predictions), axis=1)

inv_predictions = scaler.inverse_transform(inv_predictions)

inv_predictions = inv_predictions[:,1]

print(predictions[0])
print(inv_predictions[0])

# invert scaling for actual temp levels
testY = testY.reshape((len(testY), 1))
inv_temp = concatenate((unseen_X[:, -1:],testY), axis=1)
print(inv_temp[10])
inv_temp = scaler.inverse_transform(inv_temp)
inv_temp = inv_temp[:,1]

"""###predictions vs ground truth"""

plt.figure(
    figsize=(10, 7))
plt.plot(inv_temp, color='#017C8F')
plt.plot(inv_predictions, color='coral')
plt.xlabel('Time', fontsize=18)
plt.ylabel('Temperature', fontsize=18)
plt.tight_layout()
Original = mpatches.Patch(color='#017C8F', label='Original')
Forecast = mpatches.Patch(color='coral', label='Forecast')
plt.legend(handles=[Original,Forecast])
plt.show()

"""#Performance Evaluation
###model performance on unseen data


"""

# root mean squared error (RMSE)
rmse = sqrt(mean_squared_error(inv_temp, inv_predictions))
print('Test RMSE: %.3f  ' % (rmse))
print('Test RMSE: %.3f %% ' % ((rmse/max(inv_temp))*100))

# Mean absolute error
mae = keras.metrics.mean_absolute_error(inv_temp, inv_predictions)
print(mae)

import tensorflow as tf
# R
def R(y, y_pred):
  residual = tf.reduce_sum(tf.square(tf.subtract(y, y_pred)))
  total = tf.reduce_sum(tf.square(tf.subtract(y, tf.reduce_mean(y))))
  r2 = tf.subtract(1.0, tf.truediv(residual, total))
  return tf.math.sqrt(r2)

print('R score is: ', R(inv_temp, inv_predictions)) # 0.57142866

plt.figure(
    figsize=(8, 5))
plt.title("R ",fontsize=18)
plt.scatter(inv_temp, inv_predictions,color='none', marker='o', label='Data', edgecolors='black')
plt.ylabel('Predictions', fontsize=16)
plt.xlabel('Target', fontsize=18)

m, b = np.polyfit(inv_temp, inv_predictions, 1)
line2, = plt.plot(inv_temp, m*inv_temp + b,label='Fit', color='black')

first_legend = plt.legend(handles=[line2], loc='upper left',prop={'size': 15})

plt.tight_layout()
plt.show()

"""# Second approach

##padding the test set

#pre padding
"""

def generate_zeros_array(num_rows, num_columns=2):
    zeros_array = [[0] * num_columns for _ in range(num_rows)]
    return zeros_array

def create_training_data_with_pre_padding(dataset,num_observations):
  rows = math.floor(len(dataset)/138)

  zeros_arrayX = generate_zeros_array(137-num_observations, 2)

  for i in range(rows):
    concatenated_arrayX = np.concatenate((zeros_arrayX,dataset[138*i:(138*i)+num_observations,:]), axis=0)
    # concatenated_arrayX = np.concatenate((dataset[138*i:(138*i)+num_observations,:], zeros_arrayX), axis=0)
    tran_dataset.append(concatenated_arrayX)
    final_temp.append(dataset[(138*(i+1))-1,1])

num_observations = 100
final_temp = []
tran_dataset = []

num_train_points = 1398 # approx 70% of the dataset
num_val_points = 300

create_training_data_with_pre_padding(dataset, num_observations)

testX= np.array(tran_dataset[num_train_points+num_val_points:])
testY = np.array(final_temp[num_train_points+num_val_points:])

# predictions
predictions = model.predict(testX)
unseen_X = testX.reshape((testX.shape[0], 137*2))

inv_predictions = concatenate((unseen_X[:,-1:],predictions), axis=1)

inv_predictions = scaler.inverse_transform(inv_predictions)

inv_predictions = inv_predictions[:,1]

print(predictions[0])
print(inv_predictions[0])

# invert scaling for actual temp levels
testY = testY.reshape((len(testY), 1))
inv_temp = concatenate((unseen_X[:, -1:],testY), axis=1)
print(inv_temp[10])
inv_temp = scaler.inverse_transform(inv_temp)
inv_temp = inv_temp[:,1]

plt.figure(
    figsize=(10, 7))
plt.plot(inv_temp, color='#017C8F')
plt.plot(inv_predictions, color='coral')
plt.xlabel('Time', fontsize=18)
plt.ylabel('Temperature', fontsize=18)
plt.tight_layout()
Original = mpatches.Patch(color='#017C8F', label='Original')
Forecast = mpatches.Patch(color='coral', label='Forecast')
plt.legend(handles=[Original,Forecast])
plt.show()

"""## post padding"""

def create_training_data_with_post_padding(dataset,num_observations):
  rows = math.floor(len(dataset)/138)

  zeros_arrayX = generate_zeros_array(137-num_observations, 2)

  for i in range(rows):
    # concatenated_arrayX = np.concatenate((zeros_arrayX, dataset[138*i:(138*i)+num_observations,:]), axis=0)
    concatenated_arrayX = np.concatenate((dataset[138*i:(138*i)+num_observations,:], zeros_arrayX), axis=0)
    tran_dataset.append(concatenated_arrayX)
    final_temp.append(dataset[(138*(i+1))-1,1])

num_observations = 100
final_temp = []
tran_dataset = []

num_train_points = 1398 # approx 70% of the dataset
num_val_points = 300

create_training_data_with_post_padding(dataset, num_observations)

testX= np.array(tran_dataset[num_train_points+num_val_points:])
testY = np.array(final_temp[num_train_points+num_val_points:])

# predictions
predictions = model.predict(testX)
unseen_X = testX.reshape((testX.shape[0], 137*2))

inv_predictions = concatenate((unseen_X[:,-1:],predictions), axis=1)

inv_predictions = scaler.inverse_transform(inv_predictions)

inv_predictions = inv_predictions[:,1]

print(predictions[0])
print(inv_predictions[0])

# invert scaling for actual temp levels
testY = testY.reshape((len(testY), 1))
inv_temp = concatenate((unseen_X[:, -1:],testY), axis=1)
print(inv_temp[10])
inv_temp = scaler.inverse_transform(inv_temp)
inv_temp = inv_temp[:,1]

plt.figure(
    figsize=(10, 7))
plt.plot(inv_temp, color='#017C8F')
plt.plot(inv_predictions, color='coral')
plt.xlabel('Time', fontsize=18)
plt.ylabel('Temperature', fontsize=18)
plt.tight_layout()
Original = mpatches.Patch(color='#017C8F', label='Original')
Forecast = mpatches.Patch(color='coral', label='Forecast')
plt.legend(handles=[Original,Forecast])
plt.show()

"""# Third approach

##Padding both training and test
"""

num_observations = 100
final_temp = []
tran_dataset = []

num_train_points = 1398 # approx 70% of the dataset
num_val_points = 300

create_training_data_with_pre_padding(dataset, num_observations)

trainX = np.array(tran_dataset[:num_train_points])
validateX= np.array(tran_dataset[num_train_points:num_train_points+num_val_points])
testX= np.array(tran_dataset[num_train_points+num_val_points:])
trainY = np.array(final_temp[:num_train_points])
validateY = np.array(final_temp[num_train_points:num_train_points+num_val_points])
testY = np.array(final_temp[num_train_points+num_val_points:])

print(trainX.shape, validateX.shape, testX.shape, len(trainY), len(validateY), len(testY))

# print(trainX[100])

#Setup the LSTM model (with multivariate input)

pad_model = Sequential()

pad_model.add(LSTM(32, activation='relu', input_shape=(None, trainX.shape[2])))
pad_model.add(Dense(1))
pad_model.compile(loss='mse', optimizer='adam')

#training :  epochs and batch size 64
history = pad_model.fit(trainX, trainY, epochs=50, batch_size=64, callbacks=[callback],
                    validation_data=(validateX, validateY), shuffle=False)

# plot loss during training
plt.figure(
    figsize=(7, 5))
pyplot.plot(history.history['loss'], label='training')
pyplot.plot(history.history['val_loss'], label='validation')
pyplot.title('Model train vs Validation loss')
plt.xlabel('no of epochs')
plt.ylabel('loss')
pyplot.legend(['Train', 'Validation'], loc='upper right')
pyplot.show()

# model RMSE
train_score = pad_model.evaluate(trainX, trainY, verbose=1)
train_score = math.sqrt(train_score)

validation_score = pad_model.evaluate(validateX, validateY, verbose=1)
validation_score = math.sqrt(validation_score)
print('Train Score: %.2f  RMSE' % (train_score))
print('Validation Score: %.2f  RMSE' % (validation_score))

"""## padding"""

num_observations = 100
final_temp = []
tran_dataset = []

num_train_points = 1398 # approx 70% of the dataset
num_val_points = 300

create_training_data_with_pre_padding(dataset, num_observations)

testX= np.array(tran_dataset[num_train_points+num_val_points:])
testY = np.array(final_temp[num_train_points+num_val_points:])

# print(testX[0])

# predictions
predictions = pad_model.predict(testX)
unseen_X = testX.reshape((testX.shape[0], 137*2))

inv_predictions = concatenate((unseen_X[:,-1:],predictions), axis=1)

inv_predictions = scaler.inverse_transform(inv_predictions)

inv_predictions = inv_predictions[:,1]

print(predictions[0])
print(inv_predictions[0])

# invert scaling for actual temp levels
testY = testY.reshape((len(testY), 1))
inv_temp = concatenate((unseen_X[:, -1:],testY), axis=1)
print(inv_temp[10])
inv_temp = scaler.inverse_transform(inv_temp)
inv_temp = inv_temp[:,1]

plt.figure(
    figsize=(10, 7))
plt.plot(inv_temp, color='#017C8F')
plt.plot(inv_predictions, color='coral')
plt.xlabel('Time', fontsize=18)
plt.ylabel('Temperature', fontsize=18)
plt.tight_layout()
Original = mpatches.Patch(color='#017C8F', label='Original')
Forecast = mpatches.Patch(color='coral', label='Forecast')
plt.legend(handles=[Original,Forecast])
plt.show()

"""#Fourth approach (post-padding)

"""

num_observations = 100
final_temp = []
tran_dataset = []

num_train_points = 1398 # approx 70% of the dataset
num_val_points = 300

create_training_data_with_post_padding(dataset, num_observations)

trainX = np.array(tran_dataset[:num_train_points])
validateX= np.array(tran_dataset[num_train_points:num_train_points+num_val_points])
testX= np.array(tran_dataset[num_train_points+num_val_points:])
trainY = np.array(final_temp[:num_train_points])
validateY = np.array(final_temp[num_train_points:num_train_points+num_val_points])
testY = np.array(final_temp[num_train_points+num_val_points:])

print(trainX.shape, validateX.shape, testX.shape, len(trainY), len(validateY), len(testY))

print(trainY[0])

#Setup the LSTM model (with multivariate input)

pad_model = Sequential()

pad_model.add(LSTM(32, activation='relu', input_shape=(None, trainX.shape[2])))
pad_model.add(Dense(1))
pad_model.compile(loss='mse', optimizer='adam')

#training :  epochs and batch size 64
history = pad_model.fit(trainX, trainY, epochs=50, batch_size=64, callbacks=[callback],
                    validation_data=(validateX, validateY), shuffle=False)

# plot loss during training
plt.figure(
    figsize=(7, 5))
pyplot.plot(history.history['loss'], label='training')
pyplot.plot(history.history['val_loss'], label='validation')
pyplot.title('Model train vs Validation loss')
plt.xlabel('no of epochs')
plt.ylabel('loss')
pyplot.legend(['Train', 'Validation'], loc='upper right')
pyplot.show()

# model RMSE
train_score = pad_model.evaluate(trainX, trainY, verbose=1)
train_score = math.sqrt(train_score)

validation_score = pad_model.evaluate(validateX, validateY, verbose=1)
validation_score = math.sqrt(validation_score)
print('Train Score: %.2f  RMSE' % (train_score))
print('Validation Score: %.2f  RMSE' % (validation_score))

"""##observation"""

num_observations = 100
final_temp = []
tran_dataset = []

num_train_points = 1398 # approx 70% of the dataset
num_val_points = 300

create_training_data_with_post_padding(dataset, num_observations)

testX= np.array(tran_dataset[num_train_points+num_val_points:])
testY = np.array(final_temp[num_train_points+num_val_points:])

# predictions
predictions = pad_model.predict(testX)
unseen_X = testX.reshape((testX.shape[0], 137*2))

inv_predictions = concatenate((unseen_X[:,-1:],predictions), axis=1)

inv_predictions = scaler.inverse_transform(inv_predictions)

inv_predictions = inv_predictions[:,1]

print(predictions[0])
print(inv_predictions[0])

# invert scaling for actual temp levels
testY = testY.reshape((len(testY), 1))
inv_temp = concatenate((unseen_X[:, -1:],testY), axis=1)
print(inv_temp[10])
inv_temp = scaler.inverse_transform(inv_temp)
inv_temp = inv_temp[:,1]

plt.figure(
    figsize=(10, 7))
plt.plot(inv_temp, color='#017C8F')
plt.plot(inv_predictions, color='coral')
plt.xlabel('Time', fontsize=18)
plt.ylabel('Temperature', fontsize=18)
plt.tight_layout()
Original = mpatches.Patch(color='#017C8F', label='Original')
Forecast = mpatches.Patch(color='coral', label='Forecast')
plt.legend(handles=[Original,Forecast])
plt.show()

"""#Revised data"""

def create_training_data_with_pre_padding_exp(dataset):
  rows = math.floor(len(dataset)/138)

  for i in range(rows):

    for x in range(1,30):
      num_observations = 137 - x
      zeros_arrayX = generate_zeros_array(x, 2)
      concatenated_arrayX = np.concatenate((zeros_arrayX,dataset[138*i:(138*i)+num_observations,:]), axis=0)
      # concatenated_arrayX = np.concatenate((dataset[138*i:(138*i)+num_observations,:], zeros_arrayX), axis=0)
      tran_dataset.append(concatenated_arrayX)
      final_temp.append(dataset[(138*(i+1))-1,1])

final_temp = []
tran_dataset = []


create_training_data_with_pre_padding_exp(dataset)

num_train_points = int(len(tran_dataset)* 0.7) # approx 70% of the dataset
num_val_points = int(len(tran_dataset)* 0.15)

trainX = np.array(tran_dataset[:num_train_points])
validateX= np.array(tran_dataset[num_train_points:num_train_points+num_val_points])
testX= np.array(tran_dataset[num_train_points+num_val_points:])
trainY = np.array(final_temp[:num_train_points])
validateY = np.array(final_temp[num_train_points:num_train_points+num_val_points])
testY = np.array(final_temp[num_train_points+num_val_points:])

print(trainX.shape, validateX.shape, testX.shape, len(trainY), len(validateY), len(testY))

#Setup the LSTM model (with multivariate input)

new_pad_model = Sequential()

new_pad_model.add(LSTM(32, activation='relu', input_shape=(None, trainX.shape[2])))
new_pad_model.add(Dense(1))
new_pad_model.compile(loss='mse', optimizer='adam')

#training :  epochs and batch size 64
history = new_pad_model.fit(trainX, trainY, epochs=15, batch_size=64, callbacks=[callback],
                    validation_data=(validateX, validateY), shuffle=False)

# plot loss during training
plt.figure(
    figsize=(7, 5))
pyplot.plot(history.history['loss'], label='training')
pyplot.plot(history.history['val_loss'], label='validation')
pyplot.title('Model train vs Validation loss')
plt.xlabel('no of epochs')
plt.ylabel('loss')
pyplot.legend(['Train', 'Validation'], loc='upper right')
pyplot.show()

# model RMSE
train_score = new_pad_model.evaluate(trainX, trainY, verbose=1)
train_score = math.sqrt(train_score)

validation_score = new_pad_model.evaluate(validateX, validateY, verbose=1)
validation_score = math.sqrt(validation_score)
print('Train Score: %.2f  RMSE' % (train_score))
print('Validation Score: %.2f  RMSE' % (validation_score))

# predictions
predictions = new_pad_model.predict(testX)
unseen_X = testX.reshape((testX.shape[0], 137*2))

inv_predictions = concatenate((unseen_X[:,-1:],predictions), axis=1)

inv_predictions = scaler.inverse_transform(inv_predictions)

inv_predictions = inv_predictions[:,1]

# invert scaling for actual temp levels
testY = testY.reshape((len(testY), 1))
inv_temp = concatenate((unseen_X[:, -1:],testY), axis=1)
print(inv_temp[10])
inv_temp = scaler.inverse_transform(inv_temp)
inv_temp = inv_temp[:,1]

plt.figure(
    figsize=(10, 7))
plt.plot(inv_temp, color='#017C8F')
plt.plot(inv_predictions, color='coral')
plt.xlabel('Time', fontsize=18)
plt.ylabel('Temperature', fontsize=18)
plt.tight_layout()
Original = mpatches.Patch(color='#017C8F', label='Original')
Forecast = mpatches.Patch(color='coral', label='Forecast')
plt.legend(handles=[Original,Forecast])
plt.show()

"""#Visualization"""

# plt.figure(
#     figsize=(10, 7))

# plt.plot(dataset[1399*138:(1399*138)+138,1]*33, color='black')
# plt.plot(dataset[1399*138:(1399*138)+70,1]*33, color='#017C8F')
# plt.plot(137,predictions[1]*33, '--bo', color='coral')

# plt.xlabel('Time',fontsize=18)
# plt.tight_layout()
# original = mpatches.Patch(color='black', label='Original Curve')
# observed = mpatches.Patch(color='#017C8F', label='Observed')
# predicted = mpatches.Patch(color='coral', label='Predicted')
# plt.legend(handles=[original,observed,predicted])
# # positions = (0, 10000, 20000, 30000, 40000, 50000, 60000)
# # labels = ("2010", "2011", "2012", "2013", "2014", "2015", "2016")
# # plt.xticks(positions, labels)

# plt.show()

# plt.figure(
#     figsize=(10, 7))

# plt.plot(dataset[:,0], color='black')
# plt.plot(dataset[:,1], color='#017C8F')
# plt.xlabel('Time',fontsize=18)
# plt.tight_layout()
# humidity = mpatches.Patch(color='black', label='Humidity')
# temp = mpatches.Patch(color='#017C8F', label='Temperature')
# plt.legend(handles=[humidity,temp])
# # positions = (0, 10000, 20000, 30000, 40000, 50000, 60000)
# # labels = ("2010", "2011", "2012", "2013", "2014", "2015", "2016")
# # plt.xticks(positions, labels)

# plt.show()