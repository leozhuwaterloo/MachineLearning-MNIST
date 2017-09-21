import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.tensorboard.plugins import projector
import numpy as np

PATH = os.getcwd()
LOG_DIR = os.path.join(PATH, 'T-SNE_LOG')
DATA_DIR = os.path.join(PATH, 'MNIST_data')
MAX_STEPS = 10000

if __name__ == '__main__':
    if tf.gfile.Exists(LOG_DIR + '/projector'):
        tf.gfile.DeleteRecursively(LOG_DIR + '/projector')
    tf.gfile.MakeDirs(LOG_DIR + '/projector')

    mnist = input_data.read_data_sets(DATA_DIR, one_hot=True)
    metadata_path = os.path.join(LOG_DIR, 'projector/metadata.tsv')

    with open(metadata_path, 'w') as f:
        for i in range(MAX_STEPS):
            c = np.nonzero(mnist.test.labels[::1])[1:][0][i]
            f.write('{}\n'.format(c))

    sess = tf.InteractiveSession()

    with tf.device("/cpu:0"):
        images = tf.Variable(tf.stack(mnist.test.images[:MAX_STEPS], axis=0), trainable=False,
                             name='images')

    saver = tf.train.Saver()
    writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'projector'), sess.graph)
    tf.global_variables_initializer().run()

    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    embedding.tensor_name = images.name
    embedding.metadata_path = metadata_path
    embedding.sprite.image_path = os.path.join(DATA_DIR, 'mnist_10k_sprite.png')
    embedding.sprite.single_image_dim.extend([28, 28])
    projector.visualize_embeddings(writer, config)

    saver.save(sess, os.path.join(LOG_DIR, 'projector/images.ckpt'))
