import os
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=gpu,floatX=float32"

from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot
import matplotlib.patches as mpatches
from numpy import concatenate
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from numpy import save
from numpy import load
from tensorflow.keras.models import load_model
import utils
import csv
import input_formats


# Minimax scaler

# dataset = pandas.read_pickle("dataset/dataset.pkl")

# scaler  = MinMaxScaler(feature_range=(0, 1))
# dataset = scaler.fit_transform(dataset)
# final_temp, tran_dataset = utils.create_test_data(dataset,136)


################ LSTM Model

lstm_model=load_model('models/lstm_model.h5')
# lstm_model.summary()

# input formats (t' = 60)

# no padding
sample_input = input_formats.input_2D_no_padding()
LSTM_input_data_no_padding = np.array([sample_input])

print('\n=========== LSTM Model ==============\n')

print('Input shape (no padding): ',LSTM_input_data_no_padding.shape)

output = lstm_model.predict(LSTM_input_data_no_padding)

print('output: ', output[0])


# pre padding for multivariate timeseries
sample_input = input_formats.input_2D_pre_padding()
LSTM_input_data_pre_padding = np.array([sample_input])

print('\n=========== LSTM Model ==============\n')

print('Input shape (pre padding): ',LSTM_input_data_pre_padding.shape)

output = lstm_model.predict(LSTM_input_data_pre_padding)

print('output: ', output[0])


################ LSTM Model

cnn_model=load_model('models/cnn_model.h5')
# cnn_model.summary()

# input formats (t' = 60)

# pre padding for multivariate timeseries
sample_input = input_formats.input_2D_pre_padding()
CNN_input_data_pre_padding = np.array([sample_input])

print('\n=========== CNN Model ==============\n')

print('Input shape: ',CNN_input_data_pre_padding.shape)

output = cnn_model.predict(CNN_input_data_pre_padding)

print('output: ', output[0])


################ FFNN Model

dense_model = load_model('models/dense_model.h5')
# dense_model.summary()

# input formats (t' = 136)
sample_input = input_formats.input_dense()
Dense_input_data_pre_padding = np.array([sample_input])

print('\n=========== Dense Model ==============\n')

print('Input shape: ',Dense_input_data_pre_padding.shape)

output = dense_model.predict(Dense_input_data_pre_padding)

print('output: ', output[0])