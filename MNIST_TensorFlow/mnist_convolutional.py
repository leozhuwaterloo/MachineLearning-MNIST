import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


class NeutralNetwork(object):
    def __init__(self):
        with tf.name_scope('input'):
            self.x = tf.placeholder(tf.float32, [None, 784], name='x-input')
            self.y_ = tf.placeholder(tf.float32, [None, 10], name='y-input')
            self.keep_prob = tf.placeholder(tf.float32)

        with tf.name_scope('input_reshape'):
            x_image = tf.reshape(self.x, [-1, 28, 28, 1])
            tf.summary.image('input', x_image, 10)

        h_conv1 = self.deep_nn_layer(x_image, [5, 5, 1, 32], [32], 'convolutional-1', tf.nn.relu, conv2d)
        with tf.name_scope('pooling-1'):
            h_pool1 = max_pool_2x2(h_conv1)

        h_conv2 = self.deep_nn_layer(h_pool1, [5, 5, 32, 64], [64], 'convolutional-2', tf.nn.relu, conv2d)
        with tf.name_scope('pooling-2'):
            h_pool2 = max_pool_2x2(h_conv2)
            with tf.name_scope('pooling-2_flat'):
                h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])

        h_fc = self.deep_nn_layer(h_pool2_flat, [7 * 7 * 64, 1024], [1024], 'fully-connected', tf.nn.relu, tf.matmul)

        with tf.name_scope('Dropout'):
            h_fc_drop = tf.nn.dropout(h_fc, self.keep_prob)

        self.y_conv = self.deep_nn_layer(h_fc_drop, [1024, 10], [10], 'output', tf.identity, tf.matmul)

        with tf.name_scope('cross_entropy'):
            diff = tf.nn.softmax_cross_entropy_with_logits(labels=self.y_, logits=self.y_conv)
            with tf.name_scope('total'):
                self.cross_entropy = tf.reduce_mean(diff)
        tf.summary.scalar('cross_entropy', self.cross_entropy)

        with tf.name_scope('accuracy'):
            with tf.name_scope('correct_prediction'):
                correct_prediction = tf.equal(tf.argmax(self.y_conv, 1), tf.argmax(self.y_, 1))
            with tf.name_scope('accuracy'):
                self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', self.accuracy)

    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def deep_nn_layer(self, input_tensor, weight_dim, bias_dim, layer_name, act, handle):
        with tf.name_scope(layer_name):
            with tf.name_scope('weights'):
                weights = self.weight_variable(weight_dim)
                self.variable_summaries(weights)
            with tf.name_scope('biases'):
                biases = self.bias_variable(bias_dim)
                self.variable_summaries(biases)
            with tf.name_scope('Wx_plus_b'):
                preactivate = handle(input_tensor, weights) + biases
                tf.summary.histogram('pre_activations', preactivate)
            activations = act(preactivate, name='activation')
            tf.summary.histogram('activations', activations)
            return activations

    def variable_summaries(self, var):
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('histogram', var)

    def train(self, training_data, attempt_n, batch_size, learning_rate, test_data):
        if tf.gfile.Exists('ConvLogs'):
            tf.gfile.DeleteRecursively('ConvLogs')
        tf.gfile.MakeDirs('ConvLogs')

        with tf.name_scope('train'):
            train_step = tf.train.AdamOptimizer(learning_rate).minimize(self.cross_entropy)

        sess = tf.InteractiveSession()
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter('ConvLogs/train', sess.graph)
        test_writer = tf.summary.FileWriter('ConvLogs/test')
        tf.global_variables_initializer().run()

        for i in range(attempt_n):
            if i % 10 == 0:  # Record summaries and test-set accuracy
                summary, acc = sess.run([merged, self.accuracy],
                                        feed_dict={self.x: test_data.images, self.y_: test_data.labels,
                                                   self.keep_prob: 1.0})
                test_writer.add_summary(summary, i)
                print('Accuracy at step %s: %s' % (i, acc))
            else:
                batch_xs, batch_ys = training_data.next_batch(batch_size)
                if i % 100 == 99:
                    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()
                    summary, _ = sess.run([merged, train_step],
                                          feed_dict={self.x: batch_xs, self.y_: batch_ys, self.keep_prob: 0.5},
                                          options=run_options,
                                          run_metadata=run_metadata)
                    train_writer.add_run_metadata(run_metadata, 'step%d' % i)
                    train_writer.add_summary(summary, i)
                    print('Adding run metadata for', i)
                else:
                    summary, _ = sess.run([merged, train_step],
                                          feed_dict={self.x: batch_xs, self.y_: batch_ys, self.keep_prob: 0.5})
                    train_writer.add_summary(summary, i)

        train_writer.close()
        test_writer.close()
        sess.close()


if __name__ == '__main__':
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

    nn = NeutralNetwork()
    nn.train(mnist.train, 10000, 50, 1e-4, mnist.test)
