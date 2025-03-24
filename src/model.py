import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.image as mpimg

class MedicalModel(tf.keras.Model):
    def __init__(self, units, layers):
        super(MedicalModel, self).__init__()
        self.units = units
        self.dense1 = tf.keras.layers.Dense(units=self.units, activation='relu')
        self.dense2 = tf.keras.layers.Dense(units=self.units, activation='relu')
        self.conv1 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu')
        self.conv2 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu')
        self.pooling1 = tf.keras.layers.MaxPool2D(pool_size=2)
        self.pooling2 = tf.keras.layers.MaxPool2D(pool_size=2)
        self.flatten = tf.keras.layers.Flatten()
        self.dropout = tf.keras.layers.Dropout(0.2)

    def call(self, units, input):
        x = self.conv1(input)
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
        self.build(input_shape)
        self.compile(optimizer='adam', loss=self.loss)

if __name__ == '__main__':
    model = MedicalModel(units=64, layers=2)
    model = model.build_model(input_shape=(256, 256, 3))