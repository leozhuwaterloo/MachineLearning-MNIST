import gzip
import pickle

import numpy as np


def load_data():
    with gzip.open('mnist.pkl.gz', 'rb') as f:
        nPickle = pickle._Unpickler(f)
        nPickle.encoding = 'latin1'
        training_data, validation_data, test_data = nPickle.load()

    return training_data, validation_data, test_data


def load_data_wrapper():
    training_data, validation_data, test_data = load_data()
    # data are in form of numpy array of (image, label)
    # print(training_data)
    training_inputs = [np.reshape(x, (784, 1)) for x in training_data[0]]
    training_results = [vectorized_result(y) for y in training_data[1]]
    training_data = zip(training_inputs, training_results)

    validation_inputs = [np.reshape(x, (784, 1)) for x in validation_data[0]]
    validation_data = zip(validation_inputs, validation_data[1])
    test_inputs = [np.reshape(x, (784, 1)) for x in test_data[0]]
    test_data = zip(test_inputs, test_data[1])

    return list(training_data), list(validation_data), list(test_data)


def vectorized_result(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e


if __name__ == '__main__':
    training_data, validation_data, test_data = load_data_wrapper()
    print(training_data)[0]
