from sklearn.datasets import fetch_mldata
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
import numpy as np
from random import shuffle, randrange
import pickle
import os.path

mlp_file = 'saved_classifier-iterate.pkl'

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
    score_iterate=[]
    if not os.path.isfile(mlp_file):

        for a in range(1,200,1):
            mlp = MLPClassifier(hidden_layer_sizes=(31,31,31), activation='logistic', solver='lbfgs', learning_rate_init=1e-4,max_iter=a)
            print('Training neural network')
            print(mlp.max_iter)
            mlp.fit(X_train, y_train)
            output = mlp.predict(X_test)

            score = mlp.score(X_test, y_test)
            print('Accuracy: ')
            print(score)
            score_iterate.append(score)
            with open(mlp_file, 'wb') as fid:
                pickle.dump(score_iterate, fid)


    else:
        print('already trained network, delete .pkl file to retrain')
        with open(mlp_file, 'rb') as fid:
            score_iterate = pickle.load(fid)
    plt.figure()
    plt.title("influence of the number of iterations")
    plt.xlabel("Max number of iterations")
    plt.ylabel("Score")
#    plt.legend(loc="best")
    plt.plot(score_iterate)
    plt.show()
