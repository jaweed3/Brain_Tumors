import tensorflow as tf
import numpy as np
import keras
import pandas as pd
import matplotlib.image as mpimg
from keras.saving import register_keras_serializable
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # Suppress TensorFlow warnings
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # Suppress MKL duplicate library warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimizations

try:
    from keras.saving import register_keras_serializable
except ImportError:
    try:
        from tensorflow.keras.saving import register_keras_serializable
    except:
        from tensorflow.keras.utils import register_keras_serializable

@register_keras_serializable()
class MedicalModel(tf.keras.Model):
    def __init__(self, units, layers, **kwargs):
        super(MedicalModel, self).__init__(**kwargs)
        self.units = units
        self.layers_count = layers
        
        # Defining  the Layers
        self.dense1 = tf.keras.layers.Dense(units=self.units, activation='relu')
        self.dense2 = tf.keras.layers.Dense(units=self.units, activation='relu')
        self.conv1 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu')
        self.conv2 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu')
        self.pooling1 = tf.keras.layers.MaxPool2D(pool_size=2)
        self.pooling2 = tf.keras.layers.MaxPool2D(pool_size=2)
        self.flatten = tf.keras.layers.Flatten()
        self.dropout = tf.keras.layers.Dropout(0.2)

    def call(self, inputs, training=None):
        x = self.conv1(inputs)
        x = self.pooling1(x)
        x = self.conv2(x)
        x = self.pooling2(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.dense1(x)
        output = self.dense2(x)
        return output
    
    def loss(self, y_true, y_pred):
        return tf.keras.losses.mean_squared_error(y_true, y_pred)
    
    def build_model(self, input_shape):
        if len(input_shape) == 3:
            input_shape = (None,) + input_shape
        
        dummy_input = tf.keras.Input(shape=input_shape[1:])
        self(dummy_input)

        self.compile(optimizer='adam', loss=self.loss)

        return self

if __name__ == '__main__':
    model = MedicalModel(units=64, layers=2)

    model = model.build_model(input_shape=(None, 256, 256, 3))
    
    if model is not None:
        model.summary()
        print("Model built successfully!")
    else:
        print("failed to build the model!")