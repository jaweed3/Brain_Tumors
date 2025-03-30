import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.model import MedicalModel
from data.prepare_data import LoadPreprocessing

BATCH_SIZE = 32
EPOCHS = 10

class EvaluateNtrain(MedicalModel):
    def __init__(self, units=64, layers=2, **kwargs):
        super(EvaluateNtrain, self).__init__(units=units, layers=layers, **kwargs)
        self.model = MedicalModel(units=64, layers=2)
        self.preprocess = LoadPreprocessing()
        self.train_ds, self.test_ds = self.preprocess.load_dataset()
    
    def train(self):
        self.model.build_model(input_shape=(None, 256, 256, 3))
        history = self.model.fit(
            self.train_ds,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            verbose=1
        )
        return history
    
    def compile_model(self):
        self.model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy', 'Precision', 'Recall', 'AUC'],
            run_eagerly=False,
            steps_per_execution=1
        )

    def plot_model(self, history):
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

    def evaluate_model(self):
        result = self.model.evaluate(
            self.test_ds,
            batch_size=BATCH_SIZE,
            verbose=1
        )
        return result
    
if __name__ == '__main__':
    evaluate = EvaluateNtrain()
    evaluate.compile_model()
    history = evaluate.train()
    result = evaluate.evaluate_model()
    evaluate.plot_model(history)
    print(f"Evaluation result: {result}")