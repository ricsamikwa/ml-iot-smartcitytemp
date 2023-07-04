
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
from numpy import save
from numpy import load
import utils

"""# Data pre-processing"""

# fix random seed for reproducibility
np.random.seed(10)

# dataset.to_pickle("dataset/dataset.pkl")

dataset = pandas.read_pickle("dataset/dataset.pkl")
print(dataset.shape)
"""### Scaling dataset"""

# normalize the dataset (LSTMs are sensitive to the scale of the input data)
scaler  = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

dense_final_temp, dense_tran_dataset = utils.create_training_data_dense(dataset)

num_train_points = int(len(dense_tran_dataset)* 0.7) # approx 70% of the dataset
num_val_points = int(len(dense_tran_dataset)* 0.15)

# print(exp_tran_dataset[0])

trainX = np.array(dense_tran_dataset[:num_train_points])
validateX= np.array(dense_tran_dataset[num_train_points:num_train_points+num_val_points])
testX= np.array(dense_tran_dataset[num_train_points+num_val_points:])
trainY = np.array(dense_final_temp[:num_train_points])
validateY = np.array(dense_final_temp[num_train_points:num_train_points+num_val_points])
testY = np.array(dense_final_temp[num_train_points+num_val_points:])

print(trainX.shape, validateX.shape, testX.shape, len(trainY), len(validateY), len(testY))

# save to npy file ############# exp 1
save('dataset/dense_final_temp1.npy', dense_final_temp)
save('dataset/dense_tran_dataset1.npy', dense_tran_dataset)

# load the dataset

# exp_final_temp = load('dataset/exp_final_temp1.npy')
# exp_tran_dataset = load('dataset/exp_tran_dataset1.npy')