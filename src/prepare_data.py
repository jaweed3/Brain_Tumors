import requests
import os
import zipfile
import matplotlib.pyplot as plt
import tensorflow as tf

@tf.autograph.experimental.do_not_convert
class DatasetPreprocessor: 
    def __init__(self, ds_path='data', img_size=256, batch_size=32):
        self.ds_dir = ds_path
        self.img_size = img_size
        self.batch_size = batch_size
        self.train_ds = None
        self.test_ds = None
        self.class_names = None
        
        # Define preprocessing layers
        self.normalize = tf.keras.layers.Rescaling(1./255)
        self.augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomFlip('horizontal'),
            tf.keras.layers.RandomRotation(0.2),
            tf.keras.layers.RandomZoom(0.2),
            tf.keras.layers.RandomContrast(0.2)
        ])
        
    def install_ds(self, ds_url='https://www.kaggle.com/api/v1/datasets/download/masoudnickparvar/brain-tumor-mri-dataset'):
        file_name = 'dataset.zip'
        if file_name not in os.listdir():
            try:
                response = requests.get(ds_url)
                if response.status_code == 200:
                    with open(file_name, 'wb') as file:
                        file.write(response.content)
                    print("Dataset successfully downloaded")
                else:
                    print(f"Failed to download dataset. Status code: {response.status_code}")
                    return None
            except Exception as e:
                print(f"Error downloading dataset: {e}")
                return None
        else:
            print("Dataset already available")
        return file_name
    
    def unzip_ds(self, file_name):
        try:
            with zipfile.ZipFile(file_name, 'r') as file:
                file.extractall(self.ds_dir)
            print(f"Dataset successfully extracted to {self.ds_dir}")
            print(f"Contents: {os.listdir(self.ds_dir)}")
            return True
        except Exception as e:
            print(f"Error extracting dataset: {e}")
            return False
    
    def clean_dir(self):
        for file in os.listdir(self.ds_dir):
            if file != 'Training' and file != 'Testing':
                file_path = os.path.join(self.ds_dir, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    print(f"File {file} removed")
    
    def load_dataset(self):
        train_dir = os.path.join(self.ds_dir, 'Training')
        test_dir = os.path.join(self.ds_dir, 'Testing')
        
        self.train_ds = tf.keras.utils.image_dataset_from_directory(
            train_dir,
            seed=42,
            shuffle=True,
            labels='inferred',
            label_mode='int',
            verbose=True,
            image_size=(self.img_size, self.img_size),
            batch_size=self.batch_size
        )
        
        self.test_ds = tf.keras.utils.image_dataset_from_directory(
            test_dir,
            seed=42,
            shuffle=True,
            labels='inferred',
            label_mode='int',
            verbose=True,
            image_size=(self.img_size, self.img_size),
            batch_size=self.batch_size
        )
        
        self.class_names = self.train_ds.class_names
        print(f'Class names: {self.class_names}')
        
        # Optimize performance
        self.train_ds = self.train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
        self.test_ds = self.test_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
        
        return self.train_ds, self.test_ds
    
    def preprocess_image(self, image, label):
        image = tf.image.resize(image, (self.img_size, self.img_size))
        image = self.augmentation(image)
        image = self.normalize(image)
        return image, label
    
    def apply_preprocessing(self):
        if self.train_ds is None or self.test_ds is None:
            raise ValueError("Datasets not loaded. Call load_dataset() first.")
        
        self.train_ds = self.train_ds.map(
            self.preprocess_image, 
            num_parallel_calls=tf.data.AUTOTUNE
        )
        
        self.test_ds = self.test_ds.map(
            self.preprocess_image,
            num_parallel_calls=tf.data.AUTOTUNE
        )
        
        return self.train_ds, self.test_ds
    
    def plot_dataset(self, num_images=9):
        if self.train_ds is None or self.test_ds is None:
            raise ValueError("Datasets not loaded. Call load_dataset() first.")
        
        plt.figure(figsize=(12, 8))
        for images, labels in self.train_ds.take(1):
            for i in range(min(num_images, len(images))):
                ax = plt.subplot(3, 3, i + 1)
                plt.imshow(images[i].numpy().astype('uint8'))
                plt.title(f"Train: {self.class_names[labels[i]]}")
                plt.axis('off')
        plt.tight_layout()
        plt.show()
        
        plt.figure(figsize=(12, 8))
        for images, labels in self.test_ds.take(1):
            for i in range(min(num_images, len(images))):
                ax = plt.subplot(3, 3, i + 1)
                plt.imshow(images[i].numpy().astype('uint8'))
                plt.title(f"Test: {self.class_names[labels[i]]}")
                plt.axis('off')
        plt.tight_layout()
        plt.show()