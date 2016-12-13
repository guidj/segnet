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
  label = features['image/class/label'] - 1
  image = features['image/encoded']
  image = tf.cast(tf.image.decode_png(image, 3), tf.float32)
  image /= 255
  image.set_shape([224, 224, 3])
  return image, label

def inputs(train_filename, train_labels_filename, batch_size):
  image, label = read_and_decode_single_example(train_filename)
  images_batch, labels_batch = tf.train.shuffle_batch(
    [image, label], batch_size = batch_size,
    capacity=2000,
    min_after_dequeue=1000)
  return images_batch, labels_batch
