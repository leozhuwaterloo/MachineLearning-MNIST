import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from draw_graph import draw_graph_3d


class NeutralNetwork(object):
    def __init__(self, sizes, activations):
        self.x = tf.placeholder(tf.float32, [None, 784])
        self.y_ = tf.placeholder(tf.float32, [None, 10])

        layer = self.x
        for i, (from_n, to_n) in enumerate(list(zip(sizes[:-1], sizes[1:]))):
            layer = self.nn_layer(layer, from_n, to_n, activations[i])
        self.y = layer

        self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y_, logits=self.y))

        correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.y_, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def nn_layer(self, input_tensor, input_dim, output_dim, act):
        weights = self.weight_variable([input_dim, output_dim])
        biases = self.bias_variable([output_dim])
        return act(tf.matmul(input_tensor, weights) + biases)

    def train(self, training_data, attempt_n, batch_size, learning_rate, test_data):
        train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(self.cross_entropy)
        sess = tf.InteractiveSession()
        tf.global_variables_initializer().run()

        zs = []
        ys = []
        for i in range(attempt_n):
            batch_xs, batch_ys = training_data.next_batch(batch_size)
            sess.run(train_step, feed_dict={self.x: batch_xs, self.y_: batch_ys})

            if i % 10 == 0:
                acc_num = sess.run(self.accuracy,
                                   feed_dict={self.x: test_data.images, self.y_: test_data.labels})
                zs.append(acc_num)
                ys.append(i)
                print("Batch %d: %f" % (i, acc_num))

        sess.close()
        return ys, zs


if __name__ == '__main__':
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

    # nn = NeutralNetwork([784, 10], [tf.identity]) # previous model
    # learning_rates = [rate / 1000.0 for rate in range(0, 5000, 100)]
    # batch_sizes = [size for size in range(1, 11, 1)]
    neutron_ns = [n for n in range(1, 11, 1)]
    xses = []
    yses = []
    zses = []
    names = []

    for neutron_n in neutron_ns:
        print("Neutral Network with %d Neutrons" % neutron_n)
        nn = NeutralNetwork([784, neutron_n, 10], [tf.nn.relu, tf.identity])
        ys, zs = nn.train(mnist.train, 1000, 10, 0.05, mnist.test)
        xs = [neutron_n] * len(ys)
        xses.append(xs)
        yses.append(ys)
        zses.append(zs)
        names.append(str(neutron_n))

    draw_graph_3d(xses, yses, zses, names, 'Success rate with different amount of neutrons in hidden layer')
