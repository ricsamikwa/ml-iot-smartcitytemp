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
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
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

exp_final_temp = load('dataset/dense_final_temp1.npy')
exp_tran_dataset = load('dataset/dense_tran_dataset1.npy')

num_train_points = int(len(exp_tran_dataset)* 0.7) # approx 70% of the dataset
num_val_points = int(len(exp_tran_dataset)* 0.15)

trainX = np.array(exp_tran_dataset[:num_train_points])
validateX= np.array(exp_tran_dataset[num_train_points:num_train_points+num_val_points])
testX= np.array(exp_tran_dataset[num_train_points+num_val_points:])
trainY = np.array(exp_final_temp[:num_train_points])
validateY = np.array(exp_final_temp[num_train_points:num_train_points+num_val_points])
testY = np.array(exp_final_temp[num_train_points+num_val_points:])

print(trainX.shape, validateX.shape, testX.shape, len(trainY), len(validateY), len(testY))

cnn_model = Sequential()
cnn_model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(None, trainX.shape[2])))
cnn_model.add(MaxPooling1D(pool_size=2))
cnn_model.add(Conv1D(filters=16, kernel_size=2, activation='relu'))
cnn_model.add(MaxPooling1D(pool_size=2))
cnn_model.add(Flatten())
cnn_model.add(Dense(50, activation='relu'))
cnn_model.add(Dense(1))
cnn_model.compile(optimizer='adam', loss='mse')

callback = EarlyStopping(monitor='val_loss', mode='min',verbose=1,patience=50, baseline=0.000073750)

start_time = time.time()
#training :  epochs and batch size 64
history = cnn_model.fit(trainX, trainY, epochs=50, batch_size=256, callbacks=[callback],
                    validation_data=(validateX, validateY), shuffle=False)

end_time = time.time()

training_time = end_time - start_time
print("Training time: ", training_time) 

filename = 'training_time.csv'

with open('results/'+filename,'a', newline='') as file:
    writer = csv.writer(file)
    writer.writerow([5, training_time])


# model RMSE
train_score = cnn_model.evaluate(trainX, trainY, verbose=1)
train_score = math.sqrt(train_score)

validation_score = cnn_model.evaluate(validateX, validateY, verbose=1)
validation_score = math.sqrt(validation_score)
print('Train Score: %.2f  RMSE' % (train_score))
print('Validation Score: %.2f  RMSE' % (validation_score))


# save model
cnn_model.save('models/cnn_model.h5')
print('Model Saved!')
 
# load model
# pad_model=load_model('models/pad_model.h5')
# pad_model.summary()
