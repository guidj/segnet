import config
import tensorflow as tf
import utils

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
  #rgb_tensor = tf.reshape(rgb_matrix, [-1, 224, 224, 3])
  _rgb_shape = rgb_matrix.get_shape().as_list()
  _log_shape = logits.get_shape().as_list()
 
  print '[rgb][param][logits/shape: %s]' % logits.get_shape().as_list()
  print '[rgb][argmax][%s]' % argmax.get_shape().as_list()
  print '[rgb][onehotmatrix][%s]' % one_hot_matrix.get_shape().as_list()
  print '[rgb][rgb_matrix][%s]' % rgb_matrix.get_shape().as_list()
  
  rgb_tensor = tf.reshape(rgb_matrix, [-1, _log_shape[0], _log_shape[1], _rgb_shape[1]])
  return tf.cast(rgb_tensor, tf.float32)
