import tensorflow as tf

def read_and_decode_single_example(filename):
  filename_queue = tf.train.string_input_producer([filename], num_epochs=None)

  reader = tf.TFRecordReader()
  _, serialized_example = reader.read(filename_queue)

  features = tf.parse_single_example(
    serialized_example,
    features={
      'image/encoded': tf.FixedLenFeature([], tf.string)
    })

  image = features['image/encoded']
  image = tf.cast(tf.image.decode_png(image, 3), tf.float32)
  image /= 255
  #image.set_shape([224, 224, 3])
  image.set_shape([360, 480, 3])
  return image


def inputs(batch_size, train_filename, train_labels_filename=None):
  _min_after_dequeue = 16
  _capacity = _min_after_dequeue + 3 * batch_size
  image = read_and_decode_single_example(train_filename)
  _inp = [image] 

  if train_labels_filename:
    label = read_and_decode_single_example(train_labels_filename)
    _inp.append(label)

  rs = tf.train.shuffle_batch(_inp, batch_size=batch_size, capacity=_capacity, min_after_dequeue=_min_after_dequeue)

  return rs
