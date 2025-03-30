import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.image as mpimg

# Choose the correct import based on your TensorFlow/Keras version
try:
    # For newer versions
    from keras.saving import register_keras_serializable
except ImportError:
    try:
        # Alternative for newer versions
        from tensorflow.keras.saving import register_keras_serializable
    except ImportError:
        # For older versions
        from tensorflow.keras.utils import register_keras_serializable

# Define your preprocessing layer if needed
class PreProcessingLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(PreProcessingLayer, self).__init__(**kwargs)
    
    def call(self, inputs):
        # Normalize images to [0,1]
        return tf.cast(inputs, tf.float32) / 255.0

@register_keras_serializable()
class MedicalModel(tf.keras.Model):
    def __init__(self, units=64, layers=2, **kwargs):
        super(MedicalModel, self).__init__(**kwargs)
        self.units = units
        self.layers_count = layers
        
        # Define layers in init
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
        x = self.dropout(x, training=training)
        x = self.dense1(x)
        output = self.dense2(x)
        return output
    
    def loss_fn(self, y_true, y_pred):
        return tf.keras.losses.mean_squared_error(y_true, y_pred)
    
    def build_model(self, input_shape):
        # Build the model with the given input shape
        # Make sure input_shape includes batch dimension as None
        if len(input_shape) == 3:  # If batch dimension is missing
            input_shape = (None,) + input_shape
            
        # Create a dummy input to build the model
        dummy_input = tf.keras.Input(shape=input_shape[1:])
        self(dummy_input)  # This will build the model
        
        # Compile the model
        self.compile(optimizer='adam', loss=self.loss_fn)
        
        return self  # Return self for method chaining

if __name__ == '__main__':
    # Create model instance
    model = MedicalModel(units=64, layers=2)
    
    # Build and compile the model - MAKE SURE TO CAPTURE THE RETURN VALUE
    model = model.build_model(input_shape=(None, 256, 256, 3))
    
    # Verify model is not None before calling summary
    if model is not None:
        # Print model summary
        model.summary()
    else:
        print("Error: Model is None. Check the build_model method.")
