import numpy as np
import matplotlib.pyplot as plt
import urllib.request
import os
import gzip
import struct

# Константы
DATA_FOLDER = os.path.join(os.getcwd(), 'data')
URLS = {
    'train_images': 'https://azureopendatastorage.blob.core.windows.net/mnist/train-images-idx3-ubyte.gz',
    'train_labels': 'https://azureopendatastorage.blob.core.windows.net/mnist/train-labels-idx1-ubyte.gz',
    'test_images': 'https://azureopendatastorage.blob.core.windows.net/mnist/t10k-images-idx3-ubyte.gz',
    'test_labels': 'https://azureopendatastorage.blob.core.windows.net/mnist/t10k-labels-idx1-ubyte.gz'
}

# Функция для загрузки данных
def download_data():
    os.makedirs(DATA_FOLDER, exist_ok=True)
    for key, url in URLS.items():
        filename = os.path.join(DATA_FOLDER, f'{key}.gz')
        if not os.path.exists(filename):
            urllib.request.urlretrieve(url, filename)


# Функция для загрузки данных
def load_data(filename, label=False):
    with gzip.open(filename, 'rb') as gz:
        magic_number = struct.unpack('>I', gz.read(4))[0]
        n_items = struct.unpack('>I', gz.read(4))[0]
        if not label:
            n_rows = struct.unpack('>I', gz.read(4))[0]
            n_cols = struct.unpack('>I', gz.read(4))[0]
            data = np.frombuffer(gz.read(n_items * n_rows * n_cols), dtype=np.uint8)
            data = data.reshape(n_items, n_rows * n_cols)
        else:
            data = np.frombuffer(gz.read(n_items), dtype=np.uint8)
            data = data.reshape(n_items, 1)
    return data

if not os.path.exists(os.path.join(DATA_FOLDER, 'train_images.gz')):
    download_data()

# Загрузка данных
X_train = load_data(os.path.join(DATA_FOLDER, 'train_images.gz'), label=False) / 255.0
X_test = load_data(os.path.join(DATA_FOLDER, 'test_images.gz'), label=False) / 255.0
y_train = load_data(os.path.join(DATA_FOLDER, 'train_labels.gz'), label=True).reshape(-1)
y_test = load_data(os.path.join(DATA_FOLDER, 'test_labels.gz'), label=True).reshape(-1)


# Отображение примеров
count = 0
sample_size = 10
plt.figure(figsize=(16, 6))
for i in np.random.permutation(X_train.shape[0])[:sample_size]:
    count += 1
    plt.subplot(1, sample_size, count)
    plt.axhline('')
    plt.axvline('')
    plt.text(x=10, y=-10, s=y_train[i], fontsize=18)
    plt.imshow(X_train[i].reshape(28, 28), cmap=plt.cm.Greys)
plt.show()
