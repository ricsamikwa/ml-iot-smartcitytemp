import os
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"

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
from numpy import save
from numpy import load
from tensorflow.keras.models import load_model

np.random.seed(10)

exp_final_temp = load('dataset/exp_final_temp2.npy')
exp_tran_dataset = load('dataset/exp_tran_dataset2.npy')

num_train_points = int(len(exp_tran_dataset)* 0.7) # approx 70% of the dataset
num_val_points = int(len(exp_tran_dataset)* 0.15)

trainX = np.array(exp_tran_dataset[:num_train_points])
validateX= np.array(exp_tran_dataset[num_train_points:num_train_points+num_val_points])
testX= np.array(exp_tran_dataset[num_train_points+num_val_points:])
trainY = np.array(exp_final_temp[:num_train_points])
validateY = np.array(exp_final_temp[num_train_points:num_train_points+num_val_points])
testY = np.array(exp_final_temp[num_train_points+num_val_points:])

print(trainX.shape, validateX.shape, testX.shape, len(trainY), len(validateY), len(testY))


new_pad_model = Sequential()

new_pad_model.add(LSTM(32, activation='relu', input_shape=(None, trainX.shape[2])))
new_pad_model.add(Dense(1))
new_pad_model.compile(loss='mse', optimizer='adam')

callback = EarlyStopping(monitor='val_loss', mode='min',verbose=1,patience=80, baseline=0.000073750)

#training :  epochs and batch size 64
history = new_pad_model.fit(trainX, trainY, epochs=15, batch_size=64, callbacks=[callback],
                    validation_data=(validateX, validateY), shuffle=False)


# model RMSE
train_score = new_pad_model.evaluate(trainX, trainY, verbose=1)
train_score = math.sqrt(train_score)

validation_score = new_pad_model.evaluate(validateX, validateY, verbose=1)
validation_score = math.sqrt(validation_score)
print('Train Score: %.2f  RMSE' % (train_score))
print('Validation Score: %.2f  RMSE' % (validation_score))


# save model
new_pad_model.save('models/new_pad_model.h5')
print('Model Saved!')
 
# load model
# new_pad_model=load_model('models/new_pad_model.h5')
# new_pad_model.summary()
