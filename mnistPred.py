from sklearn.datasets import fetch_mldata
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
import numpy as np


def load_data(dtype=np.float32, order='F'):
    """Load the data, then cache and memmap the train/test split"""
    ######################################################################
    # Load dataset
    print("Loading dataset...")
    data = fetch_mldata('MNIST original')
    X = data['data']
    y = data["target"]

    X = X / 255

    print("Creating train-test split...")
    n_train = 60000
    X_train = X[:n_train]
    y_train = y[:n_train]
    X_test = X[n_train:]
    y_test = y[n_train:]

    return X_train, X_test, y_train, y_test



X_train, X_test, y_train, y_test = load_data()


print(X_train[:100])
print(X_train.size)
for i in range(0,10):
    ex = np.reshape(X_train[i], (28,28))
    plt.imshow(ex)
    plt.show()
