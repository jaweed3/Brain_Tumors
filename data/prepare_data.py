import  requests
import pandas as pd
import sklearn as skl
import os
import zipfile
import matplotlib.pyplot as plt
import tensorflow as tf

IMG_SIZE = 256
BATCH_SIZE = 32

class LoadPreprocessing():
    def __init__(self):
        self.ds_dir = ds_dir
        self.train_dir, self.val_dir = self.load_dataset(self.ds_dir)
        self.plot_dataset(self.train_dir, self.val_dir)
        self.preprocessing_layer = LoadPreprocessing()
        self.normalize = tf.keras.layers.Rescaling(1./255)
        self.augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomFlip('horizontal'),
            tf.keras.layers.RandomRotation(0.2),
            tf.keras.layers.RandomZoom(0.2),
            tf.keras.layers.RandomContrast(0.2),
            tf.images.random_brightness(0.2),
            tf.images.random_flip_left_right(0.2)
        ])
        self.reshape = tf.image.resize(self, (IMG_SIZE, IMG_SIZE))
    
    def install_ds(self):
        ds_url = 'https://www.kaggle.com/api/v1/datasets/download/masoudnickparvar/brain-tumor-mri-dataset'
        file_name = 'dataset.zip'
        if file_name not in os.listdir():
            try:
                response = requests.get(ds_url)
                if response.status_code == 200:
                    with open(file_name, 'wb') as file:
                        file.write(response.content)
                    print(f"Dataset berhasil diunduh dan siap digunakan")
                else:
                    print(f"Gagal mengunduh dataset.Kode Status {response.status_code}")
                    exit()
            except Exception as e:
                print(e)
        else:
            print(f"Dataset Sudah Tersedia")
        return file_name

    def unzip_ds(self, file_name, ds_dir):
        with zipfile.Zipfile(file_name, 'r') as file:
            file.extractall(ds_dir)
        print(f"Dataset telah berhasil Diextract di {ds_dir}")
        print(f"Isi dari Dataset yang telah di ekxtract: \n{os.listdir(ds_dir)}")

    def clean_dir(self, ds_dir):
        for file in os.listdir(ds_dir):
            if file != 'Training' and file != 'Testing':
                os.remove(os.path.join(ds_dir, file))
                print(f"File {file} selain data Training dan Testing telah dihapus")

    def load_dataset(self, ds_dir):
        train_dir = os.path.join(ds_dir, 'Training')
        test_dir = os.path.join(ds_dir, 'Testing')
        AUTOTUNE = tf.data.AUTOTUNE
        train_ds = tf.keras.utils.image_dataset_from_directory(
            train_dir,
            seed=42,
            shuffle=True,
            labels='inferred',
            label_mode='int',
            verbose=True,
            image_size=(IMG_SIZE, IMG_SIZE),
            batch_size=BATCH_SIZE
        )
        train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
        test_ds = tf.keras.utils.image_dataset_from_directory(
            test_dir,
            seed=42,
            shuffle=True,
            labels='inferred',
            label_mode='int',
            verbose=True,
            image_size=(IMG_SIZE, IMG_SIZE),
            batch_size=BATCH_SIZE
        )
        test_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
        print(f'Class Training Dataset: {train_dir.class_names}')
        print(f'Class Validation Dataset: {test_dir.class_names}')
        return train_ds, test_ds
    
    def preprocessing_layer(self, image, label):
        image = self.reshape(image)
        image = self.augmentation(image)
        image = self.normalize(image)
        return image, label

    def plot_dataset(self, train_ds, val_ds):
        plt.figure(figsize=(12, 8))
        for images, labels in train_ds.take(1):
            for i in range(9):
                ax = plt.subplot(3, 3, i + 1)
                plt.imshow(images[i].numpy().astype('uint8'))
                plt.title(f"Train Dataset: ", train_ds.class_names[labels[i]])
                plt.axis('off')
                plt.show()
            
        for images, labels in val_ds.take(1):
            for i in range(9):
                ax = plt.subplot(3, 3, i + 1)
                plt.imshow(images[i].numpy().astype('uint8'))
                plt.title(f"Validation Dataset: ", val_ds.class_names[labels[i]])
                plt.axis('off')
                plt.show()

    def __getitem__(self, idx):
        return self.preprocessing_layer(idx)
    
    def __len__(self):
        return len(self)

    def __call__(self, *args, **kwds):
        return self.__getitem__(*args, **kwds)
    
if __name__ == '__main__':
    ds_dir = 'data/brain_tumor_dataset'
    preprocessing = LoadPreprocessing()
    file_name = preprocessing.install_ds()
    preprocessing.unzip_ds(file_name, ds_dir)
    preprocessing.clean_dir(ds_dir)
    preprocessing.load_dataset(ds_dir)
    preprocessing.preprocessing_layer = preprocessing.preprocessing_layer
    preprocessing.train_dir = preprocessing.train_dir.map(preprocessing.preprocessing_layer)
    preprocessing.val_dir = preprocessing.val_dir.map(preprocessing.preprocessing_layer)
    preprocessing.plot_dataset(preprocessing.train_dir, preprocessing.val_dir)
    print(f"Data Preprocessing telah selesai")