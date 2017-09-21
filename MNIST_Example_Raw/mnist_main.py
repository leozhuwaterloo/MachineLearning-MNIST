import random
import numpy as np
from MNIST_Example_Raw import mnist_loader
from draw_graph import draw_graph


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))


class Network(object):
    def __init__(self, sizes, biases=None, weights=None):
        # sizes = [number of neurons in each layer]
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(output_size, 1) for output_size in sizes[1:]] if biases is None else biases
        self.weights = [np.random.randn(y, x) for (x, y) in zip(sizes[:-1], sizes[1:])] if weights is None else weights

    def feed_forward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def update_batch(self, batch, learning_rate):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        for x, y in batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        n = len(batch)
        self.weights = [w - (learning_rate / n) * nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (learning_rate / n) * nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        activation = x
        activations = [x]
        zs = []
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
                sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())

        return nabla_b, nabla_w

    def cost_derivative(self, output_activations, y):
        return (output_activations - y)

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feed_forward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def SGD(self, training_data, epochs, batch_size, learning_rate, test_data):
        train_n = len(training_data)
        test_n = len(test_data)
        graph_xs = []
        graph_ys = []
        batch_n = train_n / batch_size
        print("Batch count: %d" % batch_n)

        for i in range(epochs):
            random.shuffle(training_data)
            batches = [training_data[k: k + batch_size] for k in range(0, train_n, batch_size)]
            for j, batch in enumerate(batches):
                self.update_batch(batch, learning_rate)
            result = self.evaluate(test_data)
            graph_xs.append(i)
            graph_ys.append(result / test_n)
            print("Epoch %d: %d / %d" % (i, result, test_n))

        # draw_graph(graph_xs, graph_ys)


if __name__ == '__main__':
    random.seed(1)
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    nn = Network([784, 30, 10], np.load('biases.npy'), np.load('weights.npy'))
    nn.SGD(training_data, 30, 10, 3.0, test_data)
    np.save('biases.npy', nn.biases)
    np.save('weights.npy', nn.weights)
