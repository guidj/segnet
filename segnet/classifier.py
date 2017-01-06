import tensorflow as tf

import utils
from segnet import config

colors = tf.cast(tf.pack(utils.colors_of_dataset(config.working_dataset)), tf.float32) / 255


def color_mask(tensor, color):
    return tf.reduce_all(tf.equal(tensor, color), 3)


def one_hot(labels):
    color_tensors = tf.unstack(colors)
    channel_tensors = list(map(lambda color: color_mask(labels, color), color_tensors))
    one_hot_labels = tf.cast(tf.stack(channel_tensors, 3), 'float32')
    return one_hot_labels


def rgb(logits):
    softmax = tf.nn.softmax(logits)
    argmax = tf.argmax(softmax, 3)
    n = colors.get_shape().as_list()[0]
    one_hot = tf.one_hot(argmax, n, dtype=tf.float32)
    one_hot_matrix = tf.reshape(one_hot, [-1, n])
    rgb_matrix = tf.matmul(one_hot_matrix, colors)
    # rgb_tensor = tf.reshape(rgb_matrix, [-1, 224, 224, 3])
    _rgb_shape = rgb_matrix.get_shape().as_list()
    _log_shape = logits.get_shape().as_list()

    rgb_tensor = tf.reshape(rgb_matrix, [-1, _log_shape[1], _log_shape[2], _rgb_shape[1]])
    _rgb = tf.cast(rgb_tensor, tf.float32)

    print '[rgb][rgb][shape: %s]' % _rgb.get_shape()
    print '[rgb][softmax][shape: %s]' % softmax.get_shape()
    print '[rgb][param][logits/shape: %s]' % _log_shape
    print '[rgb][argmax][shape: %s]' % argmax.get_shape().as_list()
    print '[rgb][onehotmatrix][shape: %s]' % one_hot_matrix.get_shape().as_list()
    print '[rgb][rgb_matrix][shape: %s]' % _rgb_shape

    return _rgb
