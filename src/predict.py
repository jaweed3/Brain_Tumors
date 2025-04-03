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
    
    def load_file(self):
        root = tk.Tk()
        root.title("Tkinter Open File Dialog")
        root.resizable(True, True)
        root.geometry(300, 200)

        root.withdraw()
        file_path = fd.askopenfilename(title="Select an Image Files")
        root.destroy()
        show_info = tk.Label(root, text='File Selected', message=file_path)
        open_button = tk.Button(root, text="Open File", command=root.quit)

        open_button.pack()
        root.mainloop()
        return file_path
    
    def load_trained_model(self, model_path):
        self.model = tf.keras.models.load_model(model_path)
        return self.model
    
    def plot_selected_file(self, file_path):
        img = mpimg.imread(file_path)
        plt.figure(figsize=(10, 10))
        plt.title("Selected Image")
        plt.axis('off')
        plt.imshow(img)

    def predict(self, file_path):
        img = tf.keras.utils.load_img(file_path, target_size=(256, 256))
        img = tf.keras.utils.img_to_array(img)
        img = tf.expand_dims(img, axis=0)
        img = tf.keras.applications.resnet50.preprocess_input(img)
        prediction = self.model.predict(img)
        prediction = tf.argmax(prediction, axis=1).numpy()[0]
        class_names = ['']
        return prediction
    
if __name__ == '__main__':
    model = PredictModel(units=64, layers=2)
    model_path = 'model/medical_model.keras'
    model.load_trained_model(model_path)
    file_path = model.load_file()
    model.plot_selected_file(file_path)
    model.predict(file_path)