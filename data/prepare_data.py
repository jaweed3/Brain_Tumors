import  requests
import pandas as pd
import sklearn as skl
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
    return train_dir, val_dir

