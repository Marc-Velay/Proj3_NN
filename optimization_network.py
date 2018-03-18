from sklearn.datasets import fetch_mldata
from sklearn.neural_network import MLPClassifier
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
    X_train = X[0:36000]
    y_train = y[0:36000]
    X_test = X[36000:60000]
    y_test = y[36000:60000]
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data()
    highestScore=0
    param=(0,0,0)
    if not os.path.isfile(mlp_file):

        for a in range(1,40,5):
            for b in range(1,40,5):
                for c in range(1,40,5):
                    mlp = MLPClassifier(hidden_layer_sizes=(a, b, c), activation='logistic', solver='lbfgs', learning_rate_init=1e-4)
                    print('Training neural network')
                    print(mlp.hidden_layer_sizes)
                    mlp.fit(X_train, y_train)
                    output = mlp.predict(X_test)

                    score = mlp.score(X_test, y_test)
                    print('Accuracy: ')
                    print(score)
                    if score>highestScore:
                        highestScore = score;
                        param=(mlp.hidden_layer_sizes)
                        print("improving with param= "+str(param))
                        with open(mlp_file, 'wb') as fid:
                            pickle.dump(mlp, fid)
                    print("best is "+str(highestScore)+" with params= "+str(param))

    else:
        print('already trained network, delete .pkl file to retrain')
        with open(mlp_file, 'rb') as fid:
            mlp = pickle.load(fid)
    print(mlp.hidden_layer_sizes)
    print("Best network = "+str(mlp.score(X_test,y_test)))
