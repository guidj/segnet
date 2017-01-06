import tensorflow as tf

import classifier
import utils
import initializer
from inputs import inputs
import models
from segnet import config

train_file, train_labels_file = utils.get_training_set(config.working_dataset)

tf.app.flags.DEFINE_string('train', train_file, 'Train data')
tf.app.flags.DEFINE_string('train_ckpt', './ckpts/model.ckpt', 'Train checkpoint file')
tf.app.flags.DEFINE_string('ckpt_dir', './ckpts', 'Train checkpoint directory')
tf.app.flags.DEFINE_string('train_labels', train_labels_file, 'Train labels data')
tf.app.flags.DEFINE_string('train_logs', './logs/train', 'Log directory')
tf.app.flags.DEFINE_string('model', 'SegNetAutoencoder', 'Model to run')

tf.app.flags.DEFINE_integer('batch', 1, 'Batch size')
tf.app.flags.DEFINE_integer('steps', 50, 'Number of training iterations')


FLAGS = tf.app.flags.FLAGS


MODELS = {
    'SegNetAutoencoder': models.SegNetAutoencoder,
    'SegNetBasic': models.SegNetBasic
}


def accuracy(logits, labels):
    print '[accuracy][%s]' % logits.get_shape().as_list()
    equal_pixels = tf.reduce_sum(tf.to_float(tf.equal(logits, labels)))
    # total_pixels = FLAGS.batch * 224 * 224 * 3
    total_pixels = FLAGS.batch * 360 * 480 * 3
    return equal_pixels / total_pixels


def loss(logits, labels):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, labels)
    return tf.reduce_mean(cross_entropy, name='loss')


def train():
    print '[train][flag][batch: %s, train: %s, train_labels: %s]' % (FLAGS.batch, FLAGS.train, FLAGS.train_labels)

    try:
        images, labels = inputs(FLAGS.batch, FLAGS.train, FLAGS.train_labels)
    except tf.errors.OutOfRangeError as e:
        print e
        print '\n\nNo data found at %s' % FLAGS.train
        exit(1)

    one_hot_labels = classifier.one_hot(labels)

    print '[train][values][one_hot_labels: %s, images: %s, labels: %s]' % (
        one_hot_labels.get_shape().as_list(), images.get_shape().as_list(), labels.get_shape().as_list())

    n_classification_classes = one_hot_labels.get_shape().as_list()[-1]
    autoencoder = MODELS[FLAGS.model](n_classification_classes)
    logits = autoencoder.inference(images)

    accuracy_op = accuracy(logits, one_hot_labels)
    loss_op = loss(logits, one_hot_labels)
    tf.scalar_summary('accuracy', accuracy_op)
    tf.scalar_summary(loss_op.op.name, loss_op)

    optimizer = tf.train.AdamOptimizer(1e-04)
    train_step = optimizer.minimize(loss_op)

    init = tf.global_variables_initializer()
    # init = tf.initialize_all_variables()
    saver = tf.train.Saver()
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        ckpt = tf.train.get_checkpoint_state(FLAGS.ckpt_dir)

        if not ckpt:
            print('No checkpoint file found. Initializing...')
            global_step = 0
            sess.run(init)
            initializer.initialize(autoencoder.get_encoder_parameters(), sess)
        else:
            global_step = len(ckpt.all_model_checkpoint_paths) * FLAGS.steps
            ckpt_path = ckpt.model_checkpoint_path
            saver.restore(sess, ckpt_path)

        summary = tf.merge_all_summaries()
        summary_writer = tf.summary.FileWriter(FLAGS.train_logs,
                                               sess.graph)  # tf.train.SummaryWriter(FLAGS.train_logs, sess.graph)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        # for step in tqdm(range(FLAGS.steps + 1)):
        for step in range(FLAGS.steps + 1):
            print '[train][step][%d]' % step

            try:
                if coord.should_stop() is False:
                    sess.run(train_step)
            except tf.errors.OutOfRangeError as e:
                print e
                coord.request_stop()

            if step % 10 == 0 and coord.should_stop() is False:
                summary_str = sess.run(summary)
                summary_writer.add_summary(summary_str, step)
                summary_writer.flush()
                print '[train][step: %d][acc: %f, loss: %f]' % (step, accuracy_op.eval(), loss_op.eval())

            if step % FLAGS.batch == 0:
                saver.save(sess, FLAGS.train_ckpt, global_step=global_step)

        coord.request_stop()
        coord.join(threads)


def main(argv=None):
    utils.restore_logs(FLAGS.train_logs)
    train()


if __name__ == '__main__':
    tf.app.run()
