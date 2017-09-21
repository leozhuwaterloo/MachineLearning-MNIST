from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
from draw_graph import draw_graph


def main():
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

    x = tf.placeholder(tf.float32, [None, 784])
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    y = tf.matmul(x, W) + b

    y_ = tf.placeholder(tf.float32, [None, 10])

    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
    train_step = tf.train.GradientDescentOptimizer(0.05).minimize(cross_entropy)

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    xs = []
    ys = []
    for i in range(1500):
        batch_xs, batch_ys = mnist.train.next_batch(10)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        acc_num = sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
        xs.append(i)
        ys.append(acc_num)
        print("Batch %d: %f" % (i, acc_num))

    sess.close()
    # draw_graph(xs, ys)


if __name__ == '__main__':
    main()
