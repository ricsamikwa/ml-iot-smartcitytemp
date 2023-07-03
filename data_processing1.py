
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
print(dataset.shape)

# once !!
dataset.to_pickle("dataset/dataset.pkl")

# dataset = pandas.read_pickle("dataset/dataset.pkl")
"""### Scaling dataset"""

# normalize the dataset (LSTMs are sensitive to the scale of the input data)
scaler  = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

#number of values to be observed when making a each forecast
num_observations = 137

# frame data as supervised learning
normal_final_temp, normal_tran_dataset = utils.create_training_data(dataset, num_observations)

"""#### Spliting data to training and test set"""

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

# save to npy file ############# NORMAL
save('dataset/normal_final_temp.npy', normal_final_temp)
save('dataset/normal_tran_dataset.npy', normal_tran_dataset)

# load the dataset

# normal_final_temp = load('dataset/normal_final_temp.npy')
# normal_tran_dataset = load('dataset/normal_tran_dataset.npy')

