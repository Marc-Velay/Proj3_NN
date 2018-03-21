from sklearn.datasets import fetch_mldata
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict,ShuffleSplit
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
import numpy as np
from random import shuffle, randrange
import pickle
import os.path

mlp_file = 'saved_classifier-opt.pkl'

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


def load_data(dtype=np.float32, order='F'):
    """Load the data, then train/test split"""
    print("Loading dataset...")
    data = fetch_mldata('MNIST original')
    X = data['data']
    y = data['target']
    X = X / 255
    X, y = unison_shuffled_copies(X, y)

    print("Creating train-test split...")
    X_train = X[0:56000]#42000
    y_train = y[0:56000]
    X_test = X[56000:]
    y_test = y[56000:]

    return X_train, X_test, y_train, y_test



if __name__ == "__main__":
    X_train, X_test, y_train, y_test= load_data()
    highestScore=0
    param=(0,0,0)
    if not os.path.isfile(mlp_file):


        mlp = MLPClassifier(hidden_layer_sizes=(31,31,31), activation='logistic', solver='lbfgs', learning_rate_init=1e-4)
        mlp.fit(X_train,y_train)
        with open(mlp_file, 'wb') as fid:
            pickle.dump(mlp, fid)

    else:
        print('already trained network, delete .pkl file to retrain')
        '''with open(mlp_file, 'rb') as fid:
            mlp = pickle.load(fid)'''
        print('training MLP')
        cv = ShuffleSplit(n_splits=10, test_size=0.2)
        mlp = MLPClassifier(hidden_layer_sizes=(40, 20, 15), activation='logistic', solver='lbfgs', learning_rate_init=1e-5)
        mlp.fit(X_train, y_train)
        cross_val=cross_val_predict(mlp,X_train,y_train, n_jobs=4,verbose=9000)
        score = mlp.score(X_test, y_test)
        print('Accuracy: ')
        print(score)
        print(cross_val)
