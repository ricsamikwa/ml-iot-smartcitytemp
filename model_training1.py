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
import time
import csv

np.random.seed(10)


normal_final_temp = load('dataset/normal_final_temp.npy')
normal_tran_dataset = load('dataset/normal_tran_dataset.npy')

num_train_points = 1398 # approx 70% of the dataset
num_val_points = 300

#separated datasets
trainX = np.array(normal_tran_dataset[:num_train_points])
validateX= np.array(normal_tran_dataset[num_train_points:num_train_points+num_val_points])
testX= np.array(normal_tran_dataset[num_train_points+num_val_points:])
trainY = np.array(normal_final_temp[:num_train_points])
validateY = np.array(normal_final_temp[num_train_points:num_train_points+num_val_points])
testY = np.array(normal_final_temp[num_train_points+num_val_points:])

print(trainX.shape, validateX.shape, testX.shape, len(trainY), len(validateY), len(testY))


#Setup the LSTM model (with multivariate input)

model = Sequential()
# model.add(LSTM(32, activation='relu', input_shape=(trainX.shape[1], trainX.shape[2])))
X = keras.layers.Input(shape=(None, trainX.shape[2])) # to be added in the sequential model
model.add(LSTM(32, activation='relu', input_shape=(None, trainX.shape[2])))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')

model.summary()


callback = EarlyStopping(monitor='val_loss', mode='min',verbose=1,patience=80, baseline=0.000073750)

start_time = time.time()
#training :  epochs and batch size 64
history = model.fit(trainX, trainY, epochs=50, batch_size=256, callbacks=[callback],
                    validation_data=(validateX, validateY), shuffle=False)

end_time = time.time()

training_time = end_time - start_time
print("Training time: ", training_time) 

filename = 'training_time.csv'

with open('results/'+filename,'a', newline='') as file:
    writer = csv.writer(file)
    writer.writerow([1, training_time])
# model RMSE
train_score = model.evaluate(trainX, trainY, verbose=1)
train_score = math.sqrt(train_score)

validation_score = model.evaluate(validateX, validateY, verbose=1)
validation_score = math.sqrt(validation_score)
print('Train Score: %.2f  RMSE' % (train_score))
print('Validation Score: %.2f  RMSE' % (validation_score))


# save and load model trained


# save model
model.save('models/model.h5')
print('Model Saved!')
 
# load model
# model=load_model('models/model.h5')
# model.summary()
