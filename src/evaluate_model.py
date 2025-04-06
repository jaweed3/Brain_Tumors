import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.model import MedicalModel
from src.prepare_data import DatasetPreprocessor

BATCH_SIZE = 32
EPOCHS = 10

class EvaluateNtrain(MedicalModel):
    def __init__(self, model=None, units=64, layers=2, **kwargs):
        super(EvaluateNtrain, self).__init__(units=units, layers=layers, **kwargs)
        if model is None:
            self.model = self.model = MedicalModel(units=64, layers=2)
        else:
            self.model = model
        ds_dir = 'data/'
        self.preprocess = DatasetPreprocessor()
        self.train_ds, self.test_ds = self.preprocess.load_dataset()
    
    def train(self, train_data, val_data, epochs=EPOCHS, batch_size=BATCH_SIZE):
        self.model.build_model(input_shape=(None, 256, 256, 3))

        early_stopping = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                verbose=1,
                baseline=None,
                mode='auto',
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                verbose=1,
                mode='auto',
                min_delta=0.0001,
                min_lr=0.00001
            )
        ]

        history = self.model.fit(
            train_data,
            epochs=epochs,
            batch_size=batch_size,
            verbose=1,
            validation_data=val_data,
            callbacks=early_stopping
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

    def plot_model(self, model):
        tf.keras.utils.plot_model(
            model,
            to_file='plot_model.png',
            show_shapes=True,
            show_layer_names=True,
            rankdir='TB',
            expand_nested=True,
            show_layer_activations=True,
            show_trainable=True
        )
        print("Model plot saved as 'plot_model.png")

    def plot_learning(self, history):
        fig, axes = plt.subplots(1, 2, 2)
        axes[0].plot(history.history['loss'], label='train_loss')
        axes[0].plot(history.history['val_loss'], label='val_loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[1].plot(history.history['accuracy'], label='train_accuracy')
        axes[1].plot(history.history['val_accuracy'], label='val_accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].legend()
        plt.show()

    def evaluate_model(self):
        result = self.model.evaluate(
            self.test_ds,
            batch_size=BATCH_SIZE,
            verbose=1
        )
        return result
    
    def save_model(self, model, filepath='model.keras', model_dir='model'):
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        model.save(os.path.join(model_dir, filepath), save_format='keras')
        print(f"Model Saved at {os.path.join(model_dir, filepath)}")
    
if __name__ == '__main__':
    evaluate = EvaluateNtrain()
    evaluate.compile_model()
    history = evaluate.train()
    result = evaluate.evaluate_model()
    evaluate.plot_model(history)
    print(f"Evaluation result: {result}")