import tensorflow as tf
import math
import numpy as np

# R
def R(y, y_pred):
  residual = tf.reduce_sum(tf.square(tf.subtract(y, y_pred)))
  total = tf.reduce_sum(tf.square(tf.subtract(y, tf.reduce_mean(y))))
  r2 = tf.subtract(1.0, tf.truediv(residual, total))
  # return tf.math.sqrt(r2)
  return r2

"""#### Dataset sliding window (Multivariate)"""

def create_training_data(dataset,num_observations):

  normal_final_temp = []
  normal_tran_dataset = []
  rows = math.floor(len(dataset)/138)
  for i in range(rows):
    normal_tran_dataset.append(dataset[138*i:(138*i)+num_observations,:])
    normal_final_temp.append(dataset[(138*(i+1))-1,1])

  return normal_final_temp, normal_tran_dataset

def generate_zeros_array(num_rows, num_columns=2):
    zeros_array = [[0] * num_columns for _ in range(num_rows)]
    return zeros_array

def create_training_data_with_pre_padding_exp(dataset, exp_value = 30):
  
  exp_final_temp = []
  exp_tran_dataset = []

  rows = math.floor(len(dataset)/138)

  for i in range(rows):

    for x in range(1,exp_value):
      num_observations = 137 - x
      zeros_arrayX = generate_zeros_array(x, 2)
      concatenated_arrayX = np.concatenate((zeros_arrayX,dataset[138*i:(138*i)+num_observations,:]), axis=0)
      # concatenated_arrayX = np.concatenate((dataset[138*i:(138*i)+num_observations,:], zeros_arrayX), axis=0)
      exp_tran_dataset.append(concatenated_arrayX)
      exp_final_temp.append(dataset[(138*(i+1))-1,1])

  return exp_final_temp, exp_tran_dataset


def create_test_data(dataset,num_observations,start_row=0):

  final_temp = []
  tran_dataset = []

  rows = math.floor(len(dataset)/138)
  for i in range(start_row,rows):
    tran_dataset.append(dataset[138*i:(138*i)+num_observations,:])
    final_temp.append(dataset[(138*(i+1))-1,1])

  return final_temp, tran_dataset

def create_naive_test_data(dataset,num_observations,start_row=0):

  final_temp = []
  tran_dataset = []
  last_temp = []

  rows = math.floor(len(dataset)/138)
  for i in range(start_row,rows):
    tran_dataset.append(dataset[138*i:(138*i)+num_observations,:])
    last_temp.append(dataset[(138*i)+num_observations,1])
    final_temp.append(dataset[(138*(i+1))-1,1])

  return final_temp, tran_dataset, last_temp

def conv(value):
   return value* 5 + 0.15

def create_test_data_exp(dataset,num_observations,start_row=0):

  final_temp = []
  tran_dataset = []

  rows = math.floor(len(dataset)/138)
  # for i in range(start_row,rows):
  #   tran_dataset.append(dataset[138*i:(138*i)+num_observations,:])
  #   final_temp.append(dataset[(138*(i+1))-1,1])
  x = 137 - num_observations
  for i in range(start_row,rows):
    # for x in range(1,exp_value):
    #   num_observations = 137 - x
    zeros_arrayX = generate_zeros_array(x, 2)
    concatenated_arrayX = np.concatenate((zeros_arrayX,dataset[138*i:(138*i)+num_observations,:]), axis=0)
    # concatenated_arrayX = np.concatenate((dataset[138*i:(138*i)+num_observations,:], zeros_arrayX), axis=0)
    tran_dataset.append(concatenated_arrayX)
    final_temp.append(dataset[(138*(i+1))-1,1])

  return final_temp, tran_dataset

def get_rolling_window_bounds(start, end, window_size, step, iteration):
    """
    Get the start and end values of the rolling window for a given iteration index and step size.

    Returns:
    Tuple containing the start and end values of the current window
    """
    window_start = start + (iteration * step)
    window_end = window_start + window_size - 1
    return window_start, window_end

def flatten_inner_arrays(arr):
    flattened = []
    for item in arr:
        if isinstance(item, list):
            inner_array = []
            for inner_item in item:
                if isinstance(inner_item, list):
                    inner_array.extend(inner_item)
                else:
                    inner_array.append(inner_item)
            flattened.append(inner_array)
        else:
            flattened.append(item.flatten())
    return flattened
