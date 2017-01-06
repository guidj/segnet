import tensorflow as tf

import classifier
import convnet as cnn


# class SegNetAutoencoder:
#     def __init__(self, n, max_images=3):
#         self.params = []
#         self.n = n
#         self.max_images = max_images
#
#     def conv(self, x, channels_shape, name):
#         return cnn.conv(x, [3, 3], channels_shape, 1, name)
#
#     def deconv(self, x, channels_shape, name):
#         return cnn.deconv(x, [3, 3], channels_shape, 1, name)
#
#     def pool(self, x, size=2, stride=2):
#         return cnn.max_pool(x, size, stride)
#
#     def unpool(self, bottom):
#         sh = bottom.get_shape().as_list()
#         dim = len(sh[1:-1])
#         out = tf.reshape(bottom, [-1] + sh[-dim:])
#         for i in range(dim, 0, -1):
#             out = tf.concat(i, [out, tf.zeros_like(out)])
#         out_size = [-1] + [s * 2 for s in sh[1:-1]] + [sh[-1]]
#
#         print '[models][unpool][in/shape: %s, dim: %d, out: %s, outsize: %s]' % (
#             sh, dim, out.get_shape().as_list(), out_size)
#         return tf.reshape(out, out_size)
#
#     def encode(self, images):
#         tf.image_summary('input', images, max_images=self.max_images)
#
#         with tf.variable_scope('pool1'):
#             conv1 = self.conv(images, [3, 64], 'conv1_1')
#             conv2 = self.conv(conv1, [64, 64], 'conv1_2')
#             pool1 = self.pool(conv2, 2, 1)  # from (1, 1) to (2, 1)
#
#         with tf.variable_scope('pool2'):
#             conv3 = self.conv(pool1, [64, 128], 'conv2_1')
#             conv4 = self.conv(conv3, [128, 128], 'conv2_2')
#             pool2 = self.pool(conv4, 2, 1)
#
#         with tf.variable_scope('pool3'):
#             conv5 = self.conv(pool2, [128, 256], 'conv3_1')
#             conv6 = self.conv(conv5, [256, 256], 'conv3_2')
#             conv7 = self.conv(conv6, [256, 256], 'conv3_3')
#             pool3 = self.pool(conv7)
#
#         with tf.variable_scope('pool4'):
#             conv8 = self.conv(pool3, [256, 512], 'conv4_1')
#             conv9 = self.conv(conv8, [512, 512], 'conv4_2')
#             conv10 = self.conv(conv9, [512, 512], 'conv4_3')
#             pool4 = self.pool(conv10)
#
#         with tf.variable_scope('pool5'):
#             conv11 = self.conv(pool4, [512, 512], 'conv5_1')
#             conv12 = self.conv(conv11, [512, 512], 'conv5_2')
#             conv13 = self.conv(conv12, [512, 512], 'conv5_3')
#             pool5 = self.pool(conv13)
#
#         return pool5
#
#     def decode(self, code):
#         with tf.variable_scope('unpool1'):
#             unpool1 = self.unpool(code)
#             deconv1 = self.deconv(unpool1, [512, 512], 'deconv5_3')
#             deconv2 = self.deconv(deconv1, [512, 512], 'deconv5_2')
#             deconv3 = self.deconv(deconv2, [512, 512], 'deconv5_1')
#
#         with tf.variable_scope('unpool2'):
#             unpool2 = self.unpool(deconv3)
#             deconv4 = self.deconv(unpool2, [512, 512], 'deconv4_3')
#             deconv5 = self.deconv(deconv4, [512, 512], 'deconv4_2')
#             deconv6 = self.deconv(deconv5, [256, 512], 'deconv4_1')
#
#         with tf.variable_scope('unpool3'):
#             unpool3 = self.unpool(deconv6)
#             deconv7 = self.deconv(unpool3, [256, 256], 'deconv3_3')
#             deconv8 = self.deconv(deconv7, [256, 256], 'deconv3_2')
#             deconv9 = self.deconv(deconv8, [128, 256], 'deconv3_1')
#
#         with tf.variable_scope('unpool4'):
#             # unpool4 = self.unpool(deconv9)
#             unpool4 = deconv9
#             deconv10 = self.deconv(unpool4, [128, 128], 'deconv2_2')
#             deconv11 = self.deconv(deconv10, [64, 128], 'deconv2_1')
#
#         with tf.variable_scope('unpool5'):
#             # unpool5 = self.unpool(deconv11)
#             unpool5 = deconv11
#             deconv12 = self.deconv(unpool5, [64, 64], 'deconv1_2')
#             deconv13 = self.deconv(deconv12, [self.n, 64], 'deconv1_1')
#
#         rgb_output = classifier.rgb(deconv13)
#         tf.image_summary('output', rgb_output, max_images=self.max_images)
#
#         return deconv13
#
#     def prepare_encoder_parameters(self):
#         param_format = 'conv%d_%d_%s'
#         conv_layers = [2, 2, 3, 3, 3]
#
#         for pool in range(1, 6):
#             with tf.variable_scope('pool%d' % pool, reuse=True):
#                 for conv in range(1, conv_layers[pool - 1] + 1):
#                     weights = tf.get_variable(param_format % (pool, conv, 'W'))
#                     biases = tf.get_variable(param_format % (pool, conv, 'b'))
#                     self.params += [weights, biases]
#
#     def get_encoder_parameters(self):
#         return self.params
#
#     def inference(self, images):
#         code = self.encode(images)
#         self.prepare_encoder_parameters()
#         return self.decode(code)


class SegNetAutoencoder(object):
    def __init__(self, n, max_images=3):
        self.params = []
        self.n = n
        self.max_images = max_images

    def conv(self, x, channels_shape, name, filter_shape=None):
        if filter_shape is None:
            filter_shape = [3, 3]
        return tf.nn.relu(tf.contrib.layers.batch_norm(cnn.conv(x, filter_shape, channels_shape, 1, name)))

    def deconv(self, x, channels_shape, name, filter_shape=None):
        if filter_shape is None:
            filter_shape = [3, 3]
        return tf.nn.relu(tf.contrib.layers.batch_norm(cnn.deconv(x, filter_shape, channels_shape, 1, name)))

    def pool(self, x, size=2, stride=2):
        return cnn.max_pool(x, size, stride)

    def unpool(self, bottom):
        sh = bottom.get_shape().as_list()
        dim = len(sh[1:-1])
        out = tf.reshape(bottom, [-1] + sh[-dim:])
        for i in range(dim, 0, -1):
            out = tf.concat(i, [out, tf.zeros_like(out)])
        out_size = [-1] + [s * 2 for s in sh[1:-1]] + [sh[-1]]

        print '[models][unpool][in/shape: %s, dim: %d, out: %s, outsize: %s]' % (
            sh, dim, out.get_shape().as_list(), out_size)
        return tf.reshape(out, out_size)

    def encode(self, images):
        tf.image_summary('input', images, max_images=self.max_images)

        with tf.variable_scope('pool1'):
            conv1 = self.conv(images, [3, 64], 'conv1_1')
            conv2 = self.conv(conv1, [64, 64], 'conv1_2')
            pool1, _ = self.pool(conv2, 2, 1)  # from (1, 1) to (2, 1)

        with tf.variable_scope('pool2'):
            conv3 = self.conv(pool1, [64, 128], 'conv2_1')
            conv4 = self.conv(conv3, [128, 128], 'conv2_2')
            pool2, _ = self.pool(conv4, 2, 1)

        with tf.variable_scope('pool3'):
            conv5 = self.conv(pool2, [128, 256], 'conv3_1')
            conv6 = self.conv(conv5, [256, 256], 'conv3_2')
            conv7 = self.conv(conv6, [256, 256], 'conv3_3')
            pool3, _ = self.pool(conv7)

        with tf.variable_scope('pool4'):
            conv8 = self.conv(pool3, [256, 512], 'conv4_1')
            conv9 = self.conv(conv8, [512, 512], 'conv4_2')
            conv10 = self.conv(conv9, [512, 512], 'conv4_3')
            pool4, _ = self.pool(conv10)

        with tf.variable_scope('pool5'):
            conv11 = self.conv(pool4, [512, 512], 'conv5_1')
            conv12 = self.conv(conv11, [512, 512], 'conv5_2')
            conv13 = self.conv(conv12, [512, 512], 'conv5_3')
            pool5, _ = self.pool(conv13)

        return pool5

    def decode(self, code):
        with tf.variable_scope('unpool1'):
            unpool1 = self.unpool(code)
            deconv1 = self.deconv(unpool1, [512, 512], 'deconv5_3')
            deconv2 = self.deconv(deconv1, [512, 512], 'deconv5_2')
            deconv3 = self.deconv(deconv2, [512, 512], 'deconv5_1')

        with tf.variable_scope('unpool2'):
            unpool2 = self.unpool(deconv3)
            deconv4 = self.deconv(unpool2, [512, 512], 'deconv4_3')
            deconv5 = self.deconv(deconv4, [512, 512], 'deconv4_2')
            deconv6 = self.deconv(deconv5, [256, 512], 'deconv4_1')

        with tf.variable_scope('unpool3'):
            unpool3 = self.unpool(deconv6)
            deconv7 = self.deconv(unpool3, [256, 256], 'deconv3_3')
            deconv8 = self.deconv(deconv7, [256, 256], 'deconv3_2')
            deconv9 = self.deconv(deconv8, [128, 256], 'deconv3_1')

        with tf.variable_scope('unpool4'):
            # unpool4 = self.unpool(deconv9)
            unpool4 = deconv9
            deconv10 = self.deconv(unpool4, [128, 128], 'deconv2_2')
            deconv11 = self.deconv(deconv10, [64, 128], 'deconv2_1')

        with tf.variable_scope('unpool5'):
            # unpool5 = self.unpool(deconv11)
            unpool5 = deconv11
            deconv12 = self.deconv(unpool5, [64, 64], 'deconv1_2')
            deconv13 = self.deconv(deconv12, [self.n, 64], 'deconv1_1')

        rgb_output = classifier.rgb(deconv13)
        tf.image_summary('output', rgb_output, max_images=self.max_images)

        return deconv13

    def prepare_encoder_parameters(self):
        param_format = 'conv%d_%d_%s'
        conv_layers = [2, 2, 3, 3, 3]

        for pool in range(1, 6):
            with tf.variable_scope('pool%d' % pool, reuse=True):
                for conv in range(1, conv_layers[pool - 1] + 1):
                    weights = tf.get_variable(param_format % (pool, conv, 'W'))
                    biases = tf.get_variable(param_format % (pool, conv, 'b'))
                    self.params += [weights, biases]

    def get_encoder_parameters(self):
        return self.params

    def inference(self, images):
        code = self.encode(images)
        self.prepare_encoder_parameters()
        return self.decode(code)


class SegNetBasic(object):
    def __init__(self, n, max_images=3):
        self.params = []
        self.n = n
        self.max_images = max_images

    def conv(self, x, channels_shape, name, kernel_size=None, stride=1):
        if kernel_size is None:
            kernel_size = [7, 7]
        return tf.nn.relu(tf.contrib.layers.batch_norm(cnn.conv(x, kernel_size, channels_shape, stride, name, 'SAME')))

    def deconv(self, x, channels_shape, name, kernel_size=None, stride=1):
        # segnet-basic: upsampling, conv, batch (NO RELU)
        if kernel_size is None:
            kernel_size = [7, 7]
        return tf.contrib.layers.batch_norm(cnn.deconv(x, kernel_size, channels_shape, stride, name, 'SAME'))

    def pool(self, x, size=2, stride=2):
        return cnn.max_pool(x, size, stride)

    def pad(self, t, paddings):
        return tf.pad(t, paddings=paddings, mode='CONSTANT')

    def unpool(self, bottom):
        sh = bottom.get_shape().as_list()
        dim = len(sh[1:-1])
        out = tf.reshape(bottom, [-1] + sh[-dim:])
        for i in range(dim, 0, -1):
            out = tf.concat(i, [out, tf.zeros_like(out)])
        out_size = [-1] + [s * 2 for s in sh[1:-1]] + [sh[-1]]

        print '[models][unpool][in/shape: %s, dim: %d, out: %s, outsize: %s]' % (
            sh, dim, out.get_shape().as_list(), out_size)

        return tf.reshape(out, out_size)

    def remove_bottom_row(self, bottom):
        shape = bottom.get_shape()
        print("THIS IS THE MFUCKING SHAPE BITCH!@!@:")
        print(shape)
        out = tf.slice(bottom, [0, 0, 0, 0], [int(shape[0]), int(shape[1] - 1), int(shape[2]), int(shape[3])])
        return out

    def encode(self, images):
        paddings = [[0, 0], [0, 0], [0, 0], [0, 0]]
        tf.image_summary('input', images, max_images=self.max_images)

        with tf.variable_scope('pool1'):
            conv1 = self.conv(self.pad(images, paddings), [3, 64], 'conv1_1')
            pool1, argmax1 = self.pool(conv1)

        with tf.variable_scope('pool2'):
            conv2 = self.conv(self.pad(pool1, paddings), [64, 64], 'conv2_1')
            pool2, argmax2 = self.pool(conv2)

        with tf.variable_scope('pool3'):
            conv3 = self.conv(self.pad(pool2, paddings), [64, 64], 'conv3_1')
            pool3, argmax3 = self.pool(conv3)

        with tf.variable_scope('pool4'):
            conv4 = self.conv(self.pad(pool3, paddings), [64, 64], 'conv4_1')
            pool4, argmax4 = self.pool(conv4)

        return pool4

    def decode(self, code):
        paddings = [[0, 0], [0, 0], [0, 0], [0, 0]]
        with tf.variable_scope('unpool4'):
            unpool4 = self.unpool(code)
            unpool4 = self.remove_bottom_row(unpool4)
            deconv4 = self.deconv(self.pad(unpool4, paddings), [64, 64], 'deconv4_1')

        with tf.variable_scope('unpool3'):
            unpool3 = self.unpool(deconv4)
            deconv3 = self.deconv(self.pad(unpool3, paddings), [64, 64], 'deconv3_1')

        with tf.variable_scope('unpool2'):
            unpool2 = self.unpool(deconv3)
            deconv2 = self.deconv(self.pad(unpool2, paddings), [64, 64], 'deconv2_1')

        with tf.variable_scope('unpool1'):
            unpool1 = self.unpool(deconv2)
            deconv1 = self.deconv(self.pad(unpool1, paddings), [64, 64], 'deconv1_1')

        rgb_output = classifier.rgb(deconv1)
        tf.image_summary('output', rgb_output, max_images=self.max_images)

        # [batch, h, w, classes]
        conv_classification = tf.nn.relu(cnn.deconv(deconv1, [1, 1], [self.n, 64], 1, 'deconv1_2', 'SAME'))

        return conv_classification

        # print '[segnet-basic][decode][convclass/shape: %s]' % conv_classification.get_shape()

        # softmax = tf.nn.softmax(conv_classification)

        # return softmax

    def prepare_encoder_parameters(self):
        param_format = 'conv%d_%d_%s'
        conv_layers = [1, 1, 1, 1]

        for pool in range(1, 5):
            with tf.variable_scope('pool%d' % pool, reuse=True):
                for conv in range(1, conv_layers[pool - 1] + 1):
                    weights = tf.get_variable(param_format % (pool, conv, 'W'))
                    biases = tf.get_variable(param_format % (pool, conv, 'b'))
                    self.params += [weights, biases]

    def get_encoder_parameters(self):
        return self.params

    def inference(self, images):
        code = self.encode(images)
        self.prepare_encoder_parameters()
        return self.decode(code)
