import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


class NeutralNetwork(object):
    def __init__(self, sizes, activations, layer_names):
        with tf.name_scope('input'):
            self.x = tf.placeholder(tf.float32, [None, 784], name='x-input')
            self.y_ = tf.placeholder(tf.float32, [None, 10], name='y-input')

        with tf.name_scope('input_reshape'):
            image_shaped_input = tf.reshape(self.x, [-1, 28, 28, 1])
            tf.summary.image('input', image_shaped_input, 10)

        layer = self.x
        for i, (from_n, to_n) in enumerate(list(zip(sizes[:-1], sizes[1:]))):
            layer = self.nn_layer(layer, from_n, to_n, layer_names[i], activations[i])
        self.y = layer

        with tf.name_scope('cross_entropy'):
            diff = tf.nn.softmax_cross_entropy_with_logits(labels=self.y_, logits=self.y)
            with tf.name_scope('total'):
                self.cross_entropy = tf.reduce_mean(diff)
        tf.summary.scalar('cross_entropy', self.cross_entropy)

        with tf.name_scope('accuracy'):
            with tf.name_scope('correct_prediction'):
                correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.y_, 1))
            with tf.name_scope('accuracy'):
                self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', self.accuracy)

    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def nn_layer(self, input_tensor, input_dim, output_dim, layer_name, act):
        with tf.name_scope(layer_name):
            with tf.name_scope('weights'):
                weights = self.weight_variable([input_dim, output_dim])
                self.variable_summaries(weights)
            with tf.name_scope('biases'):
                biases = self.bias_variable([output_dim])
                self.variable_summaries(biases)
            with tf.name_scope('Wx_plus_b'):
                preactivate = tf.matmul(input_tensor, weights) + biases
                tf.summary.histogram('pre_activations', preactivate)
            activations = act(preactivate, name='activation')
            tf.summary.histogram('activations', activations)
            return activations

    def variable_summaries(self,var):
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
        if tf.gfile.Exists('Logs'):
            tf.gfile.DeleteRecursively('Logs')
        tf.gfile.MakeDirs('Logs')

        with tf.name_scope('train'):
            train_step = tf.train.AdamOptimizer(learning_rate).minimize(self.cross_entropy)

        sess = tf.InteractiveSession()
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter('Logs' + '/train', sess.graph)
        test_writer = tf.summary.FileWriter('Logs' + '/test')
        tf.global_variables_initializer().run()

        for i in range(attempt_n):
            if i % 10 == 0:  # Record summaries and test-set accuracy
                summary, acc = sess.run([merged, self.accuracy],
                                        feed_dict={self.x: test_data.images, self.y_: test_data.labels})
                test_writer.add_summary(summary, i)
                print('Accuracy at step %s: %s' % (i, acc))
            else:
                batch_xs, batch_ys = training_data.next_batch(batch_size)
                if i % 100 == 99:
                    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()
                    summary, _ = sess.run([merged, train_step],
                                          feed_dict={self.x: batch_xs, self.y_: batch_ys},
                                          options=run_options,
                                          run_metadata=run_metadata)
                    train_writer.add_run_metadata(run_metadata, 'step%d' % i)
                    train_writer.add_summary(summary, i)
                    print('Adding run metadata for', i)
                else:
                    summary, _ = sess.run([merged, train_step], feed_dict={self.x: batch_xs, self.y_: batch_ys})
                    train_writer.add_summary(summary, i)

        train_writer.close()
        test_writer.close()
        sess.close()


if __name__ == '__main__':
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

    # nn = NeutralNetwork([784, 10], [tf.identity]) # previous model
    nn = NeutralNetwork([784, 500, 10], [tf.nn.relu, tf.identity], ['layer1', 'layer2'])
    nn.train(mnist.train, 1000, 100, 0.001, mnist.test)
