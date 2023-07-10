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
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv1D, MaxPooling1D
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from numpy import save
from numpy import load
from tensorflow.keras.models import load_model
import time
import csv
import utils

np.random.seed(10)

exp_final_temp = load('dataset/exp_final_temp1.npy')
exp_tran_dataset = load('dataset/exp_tran_dataset1.npy')

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
cnn_model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(trainX.shape[1], trainX.shape[2])))
cnn_model.add(MaxPooling1D(pool_size=2))
cnn_model.add(Conv1D(filters=16, kernel_size=2, activation='relu'))
cnn_model.add(MaxPooling1D(pool_size=2))
cnn_model.add(Flatten())
cnn_model.add(Dense(50, activation='relu'))
cnn_model.add(Dense(1))
cnn_model.summary()
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
 

################################## temp I think

# dataset = pandas.read_pickle("dataset/dataset.pkl")

# scaler  = MinMaxScaler(feature_range=(0, 1))
# dataset = scaler.fit_transform(dataset)

# num_train_points = 1398 # approx 70% of the dataset
# num_val_points = 300
# start_row = num_train_points + num_val_points

# filename = 'cnn_model_evaluation2.csv'

# for n in range(1,137):

#     num_observations = n

#     print("Number of observations: " + str(num_observations))

#     final_temp, tran_dataset = utils.create_test_data_exp(dataset,num_observations,start_row)

#     iterations = 100
#     batch_size = 50

#     print_flag = 0


#     for t in range(iterations):
#       # predictions
#       start, end = utils.get_rolling_window_bounds(0, len(tran_dataset), batch_size, 2, t)
#       testX= np.array(tran_dataset[start:end])
#       testY = np.array(final_temp[start:end])
#       # predictions
#       predictions = cnn_model.predict(testX)
#       unseen_X = testX.reshape((testX.shape[0], 137*2))

#       inv_predictions = concatenate((unseen_X[:,-1:],predictions), axis=1)

#       inv_predictions = scaler.inverse_transform(inv_predictions)

#       inv_predictions = inv_predictions[:,1]


#       # invert scaling for actual temp levels
#       testY = testY.reshape((len(testY), 1))
#       inv_temp = concatenate((unseen_X[:, -1:],testY), axis=1)
    
#       inv_temp = scaler.inverse_transform(inv_temp)
#       inv_temp = inv_temp[:,1]
      
      
#       # root mean squared error (RMSE)
#       rmse = sqrt(mean_squared_error(inv_temp, inv_predictions))
#       # test_acc = 1 - (rmse/max(inv_temp))*100
      
#       # Mean absolute error
#       mae = keras.metrics.mean_absolute_error(inv_temp, inv_predictions)
      
#       R = utils.R(inv_temp, inv_predictions)

#       if print_flag == 0:
#         print_flag =2
#         print('MAE: ',mae.numpy())
#         print(inv_temp[10], inv_predictions[10])
#         print('R squared: ', R.numpy()) 
#         print('Test RMSE: %.3f  ' % (rmse))
#       # print('Test RMSE: %.3f %% ' % ((rmse/max(inv_temp))*100))

#       with open('results/'+filename,'a', newline='') as file:
#         writer = csv.writer(file)
#         writer.writerow([num_observations, rmse, mae.numpy(),R.numpy()])