from sklearn.datasets import fetch_mldata
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
import numpy as np
from numpy import transpose
from random import shuffle, randrange
import pickle
import os.path
import random
import itertools

mlp_file = 'saved_classifier_detection.pkl'

def load_data(dtype=np.float32, order='F'):
    """Load the data, then train/test split"""
    print("Loading dataset...")
    data = fetch_mldata('MNIST original')
    X = data['data']
    y = data['target']

    X = X / 255

    print("Creating train-test split...")
    n_train = 60000
    X_train = X[:n_train]
    y_train = y[:n_train]
    X_test = X[n_train:]
    y_test = y[n_train:]

    return X_train, X_test, y_train, y_test

def create_imgs(num_img):
    generated_samples = []

    for gen in range(0, num_img):
        temp = []
        sample = np.random.randint(2, size=784)
        temp.append(sample)
        #print(len(temp))
        #print(temp)
        generated_samples.append(temp[0])
    generated_samples = np.matrix(generated_samples)

    return generated_samples

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data()

    new_samples = create_imgs(30000)
    sampled_mnist = X_train[::2]
    X_train = np.concatenate((sampled_mnist, new_samples), axis=0)
    Y_train = []
    Y_train.extend(np.ones((1,30000))[0])
    Y_train.extend(np.zeros((1,30000))[0])
    Y_train = np.array(Y_train)
    Y_train = np.reshape(Y_train, (60000,1))
    X_train, Y_train = unison_shuffled_copies(X_train, Y_train)
    print(Y_train.shape)


    new_samples_test = create_imgs(10000)
    X_test = np.concatenate((X_test, new_samples_test), axis=0)
    Y_test = []
    Y_test.extend(np.ones((1,10000))[0])
    Y_test.extend(np.zeros((1,10000))[0])
    Y_test = np.array(Y_test)
    Y_test = np.reshape(Y_test, (20000,1))
    X_test, Y_test = unison_shuffled_copies(X_test, Y_test)
    print(Y_test.shape)


    if not os.path.isfile(mlp_file):
        mlp = MLPClassifier(hidden_layer_sizes=(80, 40, 15), activation='logistic', solver='lbfgs', learning_rate_init=1e-4)
        print('Training neural network')
        mlp.fit(X_train, Y_train.ravel())

        output = mlp.predict(X_test)

        score = mlp.score(X_test, Y_test.ravel())
        print('Accuracy: ')
        print(score)

        conf_mat = confusion_matrix(Y_test.ravel(), output)
        print('Confusion matrix: ')
        print(conf_mat)

        #with open(mlp_file, 'wb') as fid:
            #pickle.dump(mlp, fid)
        #    pass
    '''else:
        print('already trained network, delete .pkl file to retrain')
        with open(mlp_file, 'rb') as fid:
            mlp = pickle.load(fid)
    '''
