import  requests
import pandas as pd
import sklearn as skl
import matplotlib.pyplot as plt
import tensorflow as tf

IMG_SIZE = 256
BATCH_SIZE = 32

ds_url = ''
file_name = ''

try:
    response = requests.get(ds_url)
    if response.status_code == 200:
        with open(file_name, 'wb') as file:
            file.write(response.content)
        print(f"Dataset berhasil diunduh dan siap digunakan")
    else:
        print(f"Gagal mengunduh dataset.Kode Status {response.status_code}")
except Exception as e:
    print(e)

def load_dataset(ds_dir):
    train_dir = tf.keras.utils.image_dataset_from_directory(
        ds_dir,
        subset='training',
        seed=42,
        image_size=(IMG_SIZE, IMG_SIZE, 3),
        batch_size=BATCH_SIZE
    )
    val_dir = tf.keras.utils.image_dataset_from_directory(
        ds_dir,
        subset='validation',
        seed=42,
        image_size=(IMG_SIZE, IMG_SIZE, 3),
        batch_size=BATCH_SIZE
    )
    print(f'Class Training Dataset: {train_dir.class_names}')
    print(f'Class Validation Dataset: {val_dir.class_names}')
    return train_dir, val_dir

def plot_dataset(train_ds, val_ds):
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

def preprocess_ds(train_ds, val_ds):
    AUTOTUNE = tf.data.AUTOTUNE()