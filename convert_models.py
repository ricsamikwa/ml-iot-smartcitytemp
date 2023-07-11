import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np


################ LSTM Model

lstm_model=load_model('models/lstm_model.h5')
lstm_model.summary()

# Convert the model to TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_keras_model(lstm_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8
converter.representative_dataset = tf.lite.RepresentativeDataset.generate
tflite_model = converter.convert()
# Save the TFLite model to a file
with open('models/lstm_model.tflite', 'wb') as f:
    f.write(tflite_model)


################ CNN Model

cnn_model=load_model('models/cnn_model.h5')
cnn_model.summary()

# Convert the model to TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_keras_model(cnn_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8
converter.representative_dataset = tf.lite.RepresentativeDataset.generate
tflite_model = converter.convert()

# Save the TFLite model to a file
with open('models/cnn_model.tflite', 'wb') as f:
    f.write(tflite_model)


# ################ FFNN Model

dense_model=load_model('models/dense_model.h5')
dense_model.summary()

# Convert the model to TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_keras_model(dense_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8
converter.representative_dataset = tf.lite.RepresentativeDataset.generate
tflite_model = converter.convert()

# Save the TFLite model to a file
with open('models/dense_model.tflite', 'wb') as f:
    f.write(tflite_model)