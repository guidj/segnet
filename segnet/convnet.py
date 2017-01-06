import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_nn_ops


# https://github.com/tensorflow/tensorflow/issues/1793
# https://github.com/tensorflow/tensorflow/pull/4014
# http://stackoverflow.com/questions/39493229/how-to-use-tf-nn-max-pool-with-argmax-correctly
# maxpool/argmax is only implemented for GPU
# http://stackoverflow.com/questions/39493229/how-to-use-tf-nn-max-pool-with-argmax-correctly

@ops.RegisterGradient("MaxPoolWithArgmax")
def _MaxPoolGradWithArgmax(op, grad, unused_argmax_grad):
    return gen_nn_ops._max_pool_grad_with_argmax(op.inputs[0],
                                                 grad,
                                                 op.outputs[1],
                                                 op.get_attr("ksize"),
                                                 op.get_attr("strides"),
                                                 padding=op.get_attr("padding"))


def conv(x, receptive_field_shape, channels_shape, stride, name, padding='SAME'):
    kernel_shape = receptive_field_shape + channels_shape
    bias_shape = [channels_shape[-1]]

    weights = tf.get_variable('%s_W' % name, kernel_shape, initializer=tf.truncated_normal_initializer(stddev=.1))
    biases = tf.get_variable('%s_b' % name, bias_shape, initializer=tf.constant_initializer(.1))

    conv = tf.nn.conv2d(x, weights, strides=[1, stride, stride, 1], padding=padding)
    conv_bias = tf.nn.bias_add(conv, biases)

    print '[conv][conv2d][x/input: %s, weights/filter: %s, stride: %d, name: %s, output: %s]' % (
        x.get_shape().as_list(), weights.get_shape().as_list(), stride, name, conv_bias.get_shape())

    return conv_bias


def deconv(x, receptive_field_shape, channels_shape, stride, name, padding='SAME'):
    kernel_shape = receptive_field_shape + channels_shape
    bias_shape = [channels_shape[0]]

    input_shape = x.get_shape().as_list()
    batch_size = input_shape[0]
    height = input_shape[1]
    width = input_shape[2]

    weights = tf.get_variable('%s_W' % name, kernel_shape, initializer=tf.truncated_normal_initializer(stddev=.1))
    biases = tf.get_variable('%s_b' % name, bias_shape, initializer=tf.constant_initializer(.1))
    conv = tf.nn.conv2d_transpose(x, weights, [batch_size, height, width, channels_shape[0]], [1, stride, stride, 1],
                                  padding=padding)
    conv_bias = tf.nn.bias_add(conv, biases)
    print '[deconv][conv2d][x/input: %s, weights/filter: %s, stride: %d, name: %s, output: %s]' % (
        x.get_shape().as_list(), weights.get_shape().as_list(), stride, name, conv_bias.get_shape())

    return conv_bias


def max_pool(x, size, stride, padding='SAME'):
    # return tf.nn.max_pool(x, ksize=[1, size, size, 1], strides=[1, stride, stride, 1], padding=padding,
    #                       name='maxpool'), 1
    return tf.nn.max_pool_with_argmax(x, ksize=[1, size, size, 1], strides=[1, stride, stride, 1], padding=padding,
                                      name='maxpool')


def unravel_argmax(argmax, shape):
    output_list = [argmax // (shape[2] * shape[3]),
                   argmax % (shape[2] * shape[3]) // shape[3]]
    return tf.pack(output_list)


def unpool_layer2x2_batch(bottom, argmax):
    bottom_shape = tf.shape(bottom)
    top_shape = [bottom_shape[0], bottom_shape[1] * 2, bottom_shape[2] * 2, bottom_shape[3]]

    batch_size = top_shape[0]
    height = top_shape[1]
    width = top_shape[2]
    channels = top_shape[3]

    argmax_shape = tf.to_int64([batch_size, height, width, channels])
    argmax = unravel_argmax(argmax, argmax_shape)

    t1 = tf.to_int64(tf.range(channels))
    t1 = tf.tile(t1, [batch_size * (width // 2) * (height // 2)])
    t1 = tf.reshape(t1, [-1, channels])
    t1 = tf.transpose(t1, perm=[1, 0])
    t1 = tf.reshape(t1, [channels, batch_size, height // 2, width // 2, 1])
    t1 = tf.transpose(t1, perm=[1, 0, 2, 3, 4])

    t2 = tf.to_int64(tf.range(batch_size))
    t2 = tf.tile(t2, [channels * (width // 2) * (height // 2)])
    t2 = tf.reshape(t2, [-1, batch_size])
    t2 = tf.transpose(t2, perm=[1, 0])
    t2 = tf.reshape(t2, [batch_size, channels, height // 2, width // 2, 1])

    t3 = tf.transpose(argmax, perm=[1, 4, 2, 3, 0])

    t = tf.concat(4, [t2, t3, t1])
    indices = tf.reshape(t, [(height // 2) * (width // 2) * channels * batch_size, 4])

    x1 = tf.transpose(bottom, perm=[0, 3, 1, 2])
    values = tf.reshape(x1, [-1])

    delta = tf.SparseTensor(indices, values, tf.to_int64(top_shape))
    return tf.sparse_tensor_to_dense(tf.sparse_reorder(delta))
