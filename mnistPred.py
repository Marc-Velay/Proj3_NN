from sklearn.datasets import fetch_mldata
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
import numpy as np
from random import shuffle, randrange
import pickle
import os.path

mlp_file = 'saved_classifier.pkl'

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
    X_train = X[0:42000]
    y_train = y[0:42000]
    X_test = X[42000:60000]
    y_test = y[42000:60000]
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data()


    '''for i in range(0,5):
        ex = np.reshape(X_train[i*10000], (28,28))
        plt.imshow(ex)
        plt.show()
        print(y_train[i*10000])
        '''
    if not os.path.isfile(mlp_file):
        mlp = MLPClassifier(hidden_layer_sizes=(80, 40, 15), activation='logistic', solver='lbfgs', learning_rate_init=1e-4)
        print('Training neural network')
        mlp.fit(X_train, y_train)

        output = mlp.predict(X_test)

        score = mlp.score(X_test, y_test)
        print('Accuracy: ')
        print(score)

        conf_mat = confusion_matrix(y_test, output)
        print('Confusion matrix: ')
        print(conf_mat)

        with open(mlp_file, 'wb') as fid:
            pickle.dump(mlp, fid)
    else:
        print('already trained network, delete .pkl file to retrain')
        with open(mlp_file, 'rb') as fid:
            mlp = pickle.load(fid)

    for i in range(0, 10):
        index = randrange(0, 36000, 1)
        pred = mlp.predict(X_train[index:index+1])
        pred_proba = mlp.predict_proba(X_train[index:index+1])
        print("We predict: ")
        print(pred[0])
        print("Was supposed to be: ")
        print(y_train[index])
        print("Proba: ")
        print(pred_proba)
        print()

        f0 = plt.figure(1)
        plt.bar(np.arange(10), pred_proba[0])
        f0.show()

        f1 = plt.figure(2)
        ex = np.reshape(X_train[index:index+1], (28,28))
        plt.imshow(ex, cmap='gray')
        f1.show()
        print('press enter to view next example: ')
        #input()
        plt.close("all")


    score = mlp.score(X_test, y_test)
    print('Accuracy: ')
    print(score)
