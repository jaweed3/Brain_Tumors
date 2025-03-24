import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from model import MedicalModel

BATCH_SIZE = 32
EPOCHS = 10

class EvaluateNtrain(MedicalModel):
    def __init__(self):
        self.model = MedicalModel(units=64, layers=2)
    
    def train(self.model):
        history = self.model.fit(
            x_train,
            y_train,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_data=(x_val, y_val),
            verbose=1
        )
        return history

    def plot_model(history):
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs = range(1, len(loss) + 1)
        plt.figure(figsize=(12, 8))
        plt.plot(epochs, loss, 'bo', label='Training loss')
        plt.plot(epochs, val_loss, 'b', label='Validation Loss')
        plt.title('Training and Validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

    def evaluate_model(self.model, history):
        history = self.model.evaluate(
            x_test,
            y_test,
            batch_size=BATCH_SIZE,
            verbose=1
        )
        self.plot_model(history)
        return history