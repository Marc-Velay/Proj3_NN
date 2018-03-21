from sklearn.datasets import fetch_mldata
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
import numpy as np
from random import shuffle, randrange
import pickle
import os.path
from numpy import transpose

mlp_file_estimation = 'saved_classifier-estimation.pkl'
mlp_file = 'saved_classifier.pkl'


def load_data(dtype=np.float32, order='F'):
    """Load the data, then train/test split"""
    print("Loading dataset...")
    data = fetch_mldata('MNIST original')
    X = data['data']
    y = data['target']
    X = X / 255

    print("Creating train-test split...")
    X_train = X[0:60000]
    y_train = y[0:60000]
    X_test = X[42000:60000]
    y_test = y[42000:60000]
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    data = fetch_mldata('MNIST original')
    X = data['data']
    Y = data['target']
    counter = np.zeros((10,1))
    for i in range(0, len(Y)):
        counter[int(Y[i])]+=1
    counter/=len(Y)
    counter*=100
    plt.bar(np.arange(10),counter.transpose()[0],color='b',width=0.8,label="estimation training set")


    X_train, X_test, y_train, y_test = load_data()
    if not os.path.isfile(mlp_file):

        mlp = MLPClassifier(hidden_layer_sizes=(80, 40, 15), activation='logistic', solver='lbfgs', learning_rate_init=1e-4)
        print('Training neural network')
        mlp.fit(X_train, y_train)
        with open(mlp_file, 'wb') as fid:
            pickle.dump(mlp, fid)

    else:
        print('already trained network, delete .pkl file to retrain')
        with open(mlp_file, 'rb') as fid:
            mlp = pickle.load(fid)
    output=mlp.predict(X)
    counter_predidt = np.zeros((10,1))
    for i in range(0, len(output)):
        counter_predidt[int(output[i])]+=1
    counter_predidt/=len(output)
    counter_predidt*=100
    plt.bar(np.arange(10),counter_predidt.transpose()[0],width=0.4,color='r',label="estimation prediction")
    plt.legend()
    plt.show()
