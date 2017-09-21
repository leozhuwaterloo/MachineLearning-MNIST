import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from PIL import Image
import numpy as np
import argparse
import sys


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


class NeutralNetwork(object):
    def __init__(self):
        self.x = tf.placeholder(tf.float32, [None, 784], name='x-input')
        self.y_ = tf.placeholder(tf.float32, [None, 10], name='y-input')
        self.keep_prob = tf.placeholder(tf.float32)

        x_image = tf.reshape(self.x, [-1, 28, 28, 1])

        h_conv1 = self.deep_nn_layer(x_image, [5, 5, 1, 32], [32], tf.nn.relu, conv2d)
        h_pool1 = max_pool_2x2(h_conv1)

        h_conv2 = self.deep_nn_layer(h_pool1, [5, 5, 32, 64], [64], tf.nn.relu, conv2d)
        h_pool2 = max_pool_2x2(h_conv2)
        h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])

        h_fc = self.deep_nn_layer(h_pool2_flat, [7 * 7 * 64, 1024], [1024], tf.nn.relu, tf.matmul)
        h_fc_drop = tf.nn.dropout(h_fc, self.keep_prob)

        self.y_conv = self.deep_nn_layer(h_fc_drop, [1024, 10], [10], tf.identity, tf.matmul)

        self.cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=self.y_, logits=self.y_conv))

        correct_prediction = tf.equal(tf.argmax(self.y_conv, 1), tf.argmax(self.y_, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def deep_nn_layer(self, input_tensor, weight_dim, bias_dim, act, handle):
        weights = self.weight_variable(weight_dim)
        biases = self.bias_variable(bias_dim)
        return act(handle(input_tensor, weights) + biases)

    def train(self, training_data, attempt_n, batch_size, learning_rate, test_data):
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(self.cross_entropy)

        sess = tf.InteractiveSession()
        tf.global_variables_initializer().run()
        saver = tf.train.Saver()

        for i in range(attempt_n):
            batch_xs, batch_ys = training_data.next_batch(batch_size)
            if i % 100 == 99:
                print("Step %d: %f" % (i, sess.run(self.accuracy,
                                                   feed_dict={self.x: test_data.images, self.y_: test_data.labels,
                                                              self.keep_prob: 1.0})))
                saver.save(sess, 'Conv_Model/tmp_mnist_conv.ckpt')
            sess.run(train_step,
                     feed_dict={self.x: batch_xs, self.y_: batch_ys, self.keep_prob: 0.5})
        sess.close()

    def predict(self, path, test_img):
        sess = tf.InteractiveSession()
        saver = tf.train.Saver()
        saver.restore(sess, path)

        result = tf.argmax(self.y_conv, 1)
        print("Result: %f" % sess.run(result,
                                      feed_dict={self.x: test_img, self.keep_prob: 1.0}))
        sess.close()


if __name__ == '__main__':
    # mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

    with Image.open('download.png') as img:
        img.thumbnail((28, 28))
        img = img.convert('L')
        img_data = np.asarray(img, dtype=np.float32)
        img_data = img_data.reshape([784, 1])
        final_data = []
        for pixel in img_data:
            alpha = pixel[0]
            final_data.append((255.0 - alpha) / 255.0)
        print(final_data)
        nn = NeutralNetwork()
        nn.predict('Conv_Model/mnist_conv_model.ckpt', [final_data])
        # nn.train(mnist.train, 20000, 50, 1e-4, mnist.test)
