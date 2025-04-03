import tensorflow as tf
import matplotlib.pyplot as plt
import os
import matplotlib.image as mpimg
import tkinter as tk
from tkinter import filedialog as fd
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model import MedicalModel

class PredictModel(MedicalModel):
    def __init__(self, units, layers, **kwargs):
        super().__init__(units, layers, **kwargs)
        self.model = None
        self.class_name = ['glioma', 'meningioma', 'pituitary tumor', 'no tumor']
        self.file_path = None

    def load_file(self):
        '''
        Trying to open load file dialog using tkinter
        '''
        try:

            root = tk.Tk()
            root.title("Tkinter Open File Dialog")
            root.resizable(True, True)
            root.geometry(300, 200)

            root.withdraw()
            file_path = fd.askopenfilename(
                title="Select an Image Files",
                filetypes=[
                    ("Image Files", '*.png', '*.jpg', '*.jpeg'),
                    ("All Files", '*.*')
                ])
            root.destroy()

            if not file_path:
                raise ValueError("No File Selected")
            
            return file_path
        
        except Exception as e:
            print('f"Error Loading File: {e}')
            return None
    
    def load_trained_model(self, model_path):
        '''
        Loading model with specified path
        '''
        try:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
            self.model = tf.keras.models.load_model(model_path)
            print(f"Model loaded from {model_path}")
            return self.model
        except Exception as e:
            print(f"Error loading model: {e}")
            return None

    def plot_selected_file(self, file_path):
        '''
        Plotting the selected file using matplotlib image library
        '''
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File Not Found: {file_path}")
            img = mpimg.imread(file_path)
            plt.figure(figsize=(10, 10))
            plt.title("Selected Image")
            plt.axis('off')
            plt.imshow(img)
            plt.show()
        except Exception as e:
            print(f"Error When Plottting the Image: {e}")

    def predict(self, file_path):
        '''
        Predicting the selected class with loaded model before
        '''
        try:
            if not file_path or not os.path.exists(file_path):
                raise ValueError("File Path is not Valid or The File Does Not Exist")
            if self.model is None:
                raise ValueError("Model is not loaded.Please load the model first")
            
            # Load and Preprocess the selected image
            img = tf.keras.utils.load_img(file_path, target_size=(256, 256))
            img = tf.keras.utils.img_to_array(img)
            img = tf.expand_dims(img, axis=0)
            img = tf.keras.applications.resnet50.preprocess_input(img)

            # Make Prediction
            prediction = self.model.predict(img)
            prediction_class_idx = tf.argmax(prediction, axis=1).numpy()[0]
            
                
if __name__ == '__main__':
    model = PredictModel(units=64, layers=2)
    model_path = 'model/medical_model.keras'
    model.load_trained_model(model_path)
    file_path = model.load_file()
    model.plot_selected_file(file_path)
    model.predict(file_path)