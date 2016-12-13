from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_nn_ops

import tensorflow as tf

def upsample(bottom):
  sh = bottom.get_shape().as_list()
  dim = len(sh[1:-1])
  out = tf.reshape(bottom, [-1] + sh[-dim:])
  for i in range(dim, 0, -1):
    out = tf.concat(i, [out, tf.zeros_like(out)])
  out_size = [-1] + [s * 2 for s in sh[1:-1]] + [sh[-1]]
  out = tf.reshape(out, out_size)
return out
