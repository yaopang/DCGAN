import numpy as np
import tensorflow as tf
# import cv2
import math
import os
# import yaml
import time
import scipy.misc
import datetime
import sys
from scipy.ndimage.filters import gaussian_filter

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# tf.logging.set_verbosity(tf.logging.INFO)


class colors:
    header = '\033[95m'
    blue = '\033[94m'
    green = '\033[92m'
    yellow = '\033[93m'
    red = '\033[91m'
    endc = '\033[0m'
    bold = '\033[1m'
    underline = '\033[4m'


def center_crop(data, desired_shape):
    start_crop = (np.array(data.shape[1:3]) - desired_shape[:2])//2
    return data[:,
                start_crop[0]:start_crop[0]+desired_shape[0],
                start_crop[1]:start_crop[1]+desired_shape[1]]


def make_batches(dataset, batch_size):
    for i in range(0, dataset.shape[0], batch_size):
        yield dataset[i:i + batch_size]


def leaky_relu(layer, alpha):
    return tf.maximum(layer, alpha*layer)


def conv_lrelu_layer(input,
                     filter,
                     bias,
                     strides,
                     padding,
                     alpha,
                     do_batch_norm=False,
                     is_training=True,
                     momentum=.9,
                     epsilon=1e-5):

    layer = tf.nn.conv2d(input=input, filter=filter, strides=strides, padding=padding)
    layer = tf.reshape(tf.nn.bias_add(layer, bias), layer.get_shape())
    if do_batch_norm:
        layer = tf.contrib.layers.batch_norm(layer,
                                             decay=momentum,
                                             updates_collections=None,
                                             epsilon=epsilon,
                                             scale=True,
                                             is_training=is_training)
    return leaky_relu(layer, alpha)


def conv_relu_transpose_layer(input,
                              filter,
                              bias,
                              strides,
                              output_shape,
                              padding,
                              do_relu=True,
                              do_batch_norm=False,
                              is_training=True,
                              momentum=.9,
                              epsilon=1e-5):
    layer = tf.nn.conv2d_transpose(value=input, filter=filter, output_shape=output_shape, strides=strides, padding=padding)
    layer = tf.reshape(tf.nn.bias_add(layer, bias), layer.get_shape())
    if do_batch_norm:
        layer = tf.contrib.layers.batch_norm(layer,
                                             decay=momentum,
                                             updates_collections=None,
                                             epsilon=epsilon,
                                             scale=True,
                                             is_training=is_training)
    if do_relu:
        return tf.nn.relu(layer)
    else:
        return layer


def conv_output_shape(input_shape, stride):
    if type(input_shape) is not list:
        input_shape = input_shape.tolist()
    return [int(math.ceil(float(size) / float(stride))) if i < 2 else int(size) for i, size in enumerate(input_shape)]


def visualizer_shape(num_images):
    h = int(np.floor(np.sqrt(num_images)))
    w = int(np.ceil(np.sqrt(num_images)))
    assert h * w == num_images
    return [h, w]


def visualizer_merge(images, size):
    h, w = images.shape[1], images.shape[2]
    c = images.shape[3]
    img = np.zeros((h * size[0], w * size[1], c), dtype=np.uint8)
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j * h:j * h + h, i * w:i * w + w, :] = image
    return img


def save_samples(images, path, name):
    vis_shape = visualizer_shape(images.shape[0])
    images = np.uint8(255.*(images + 1.)/2.)
    vis_image = visualizer_merge(images, vis_shape)
    # cv2.imwrite(os.path.join(path, name), cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
    scipy.misc.imsave(os.path.join(path, name), vis_image)


def save_individual_samples(images, path, name):
    images = np.uint8(255.*(images + 1.)/2.)
    name_split = name.split('.')
    for i in range(images.shape[0]):
        new_name = name_split[0] + str(i) + '.' + name_split[1]
        scipy.misc.imsave(os.path.join(path, new_name), images[i])


class DCGAN:
    def __init__(self,
                 session,
                 data,
                 input_shape,
                 output_shape,
                 crop_before_resize=True,
                 learning_rate=0.0002,
                 epsilon=1e-5,
                 beta1=0.5,
                 momentum=0.9,
                 batch_size=256,
                 g_feature_dim=64,
                 d_feature_dim=64,
                 z_dim=100,
                 leaky_relu_alpha=.2,
                 kernel_size=5,
                 kernel_init_stddev=.02,
                 bias_init=0.0,
                 log_dir='./logs',
                 visualize=True,
                 save_on_epoch=True,
                 save_epoch=25,
                 num_sample=64,
                 sample_epoch=10,
                 verbose=True):

        self.session = session
        self.train_data = data
        self.input_shape = input_shape
        self.output_shape = output_shape

        assert len(data.shape) == 4
        assert len(input_shape) == 3
        assert len(output_shape) == 3

        self.crop_before_resize = crop_before_resize
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.momentum = momentum
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.visualize = visualize
        self.g_feature_dim = g_feature_dim
        self.d_feature_dim = d_feature_dim
        self.z_dim = z_dim
        self.leaky_relu_alpha = leaky_relu_alpha
        self.kernel_size = kernel_size
        self.kernel_init_stddev = kernel_init_stddev
        self.bias_init = bias_init
        self.log_dir = log_dir
        self.verbose = verbose
        self.num_sample = num_sample
        self.save_on_epoch = save_on_epoch
        self.save_epoch = save_epoch
        self.sample_epoch = sample_epoch

        self.d_losses = []
        self.g_losses = []

        self.save_dir = self.get_save_dir()
        self.samples_dir = os.path.join(self.save_dir, 'samples')

        self.config = {"input_shape": self.input_shape,
                       "output_shape": self.output_shape,
                       "crop_before_resize": self.crop_before_resize,
                       "learning_rate": self.learning_rate,
                       "epsilon": self.epsilon,
                       "beta1": self.beta1,
                       "momentum": self.momentum,
                       "batch_size": self.batch_size,
                       "g_feature_dim": self.g_feature_dim,
                       "d_feature_dim": self.d_feature_dim,
                       "z_dim": self.z_dim,
                       "leaky_relu_alpha": self.leaky_relu_alpha,
                       "kernel_size": self.kernel_size,
                       "kernel_init_stddev": self.kernel_init_stddev,
                       "bias_init": self.bias_init,
                       "log_dir": self.log_dir,
                       "save_dir": self.save_dir,
                       "visualize": self.visualize,
                       "num_sample": self.num_sample,
                       "save_on_epoch": self.save_on_epoch,
                       "save_epoch": self.save_epoch,
                       "verbose": self.verbose}

        # self.write_config()
        self.write_config_primitive()
        if verbose:
            self.print_config()

        self.transform_data()

        batches = list(make_batches(self.train_data, self.batch_size))
        self.batches = [batch for batch in batches if batch.shape[0] == self.batch_size]

        self.build_model()

    def print_config(self):
        print('Config:')
        for key, value in self.config.items():
            print ('    ' + key + ': ' + str(value))
        print('\n')

    def get_save_dir(self):
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        save_dir = self.log_dir + '/run_at_' + str(datetime.datetime.now()).replace(' ', '_')[:-7]
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # for i in range(1000):
        #     save_dir = self.log_dir + '/run' + str(i) + '/'
        #     if os.path.exists(save_dir) and os.listdir(save_dir) == []:
        #         break
        #     if not os.path.exists(save_dir):
        #         os.makedirs(save_dir)
        #         break

        if self.verbose:
            print('\ncreated save directory at ' + str(save_dir))

        return save_dir

    def transform_data(self):
        if self.verbose:
            print('transforming training data...')

        original_shape = self.train_data.shape
        if self.crop_before_resize:
            self.train_data = center_crop(self.train_data, self.input_shape)

        if not self.train_data.shape[1:] == tuple(self.output_shape):
            resized_data = np.zeros([self.train_data.shape[0]] + self.output_shape)
            for i in range(self.train_data.shape[0]):
                # resized_data[i] = cv2.resize(data[i], tuple(self.output_shape[:2]))
                resized_data[i] = scipy.misc.imresize(self.train_data[i], self.output_shape[:2])

            is_uint8 = np.max(resized_data[0]) >= 127 and np.min(resized_data[0]) >= 0
            if is_uint8:
                resized_data = resized_data/127.5 - 1.
            else:
                resized_data = 2.*resized_data - 1.

            self.train_data = resized_data

        if self.verbose:
            print('transformed training data from ' + str(original_shape) + ' to ' + str(self.train_data.shape))

    # def write_config(self):
    #     with open(self.save_dir + 'run_config.yml', 'w') as outfile:
    #         yaml.dump(self.config, outfile, default_flow_style=False)

    def write_config_primitive(self):
        with open(os.path.join(self.save_dir, 'run_config.yml'), 'w') as f:
            for key, value in self.config.items():
                if type(value) is not list:
                    f.write(key + ': ' + str(value) + '\n')
                else:
                    f.write(key + '\n')
                    for val in value:
                        f.write('- ' + str(val) + '\n')

    def build_model(self):
        if self.verbose:
            print ('building model...')

        self.inputs = tf.placeholder(tf.float32, [self.batch_size] + self.output_shape, name='training_data')
        self.d_logits_real = self.discriminator(self.inputs, is_training=True, reuse=False)

        self.z = tf.placeholder(tf.float32, [None, self.z_dim], name='Z')
        self.g_logits = self.generator(self.z, is_training=True, reuse=False)
        self.d_logits_fake = self.discriminator(self.g_logits, is_training=True, reuse=True)

        with tf.name_scope('d_loss_real'):
            self.d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_logits_real,
                                                                                      labels=tf.ones_like(self.d_logits_real, name='d_ones')))
        with tf.name_scope('g_loss'):
            self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_logits_fake,
                                                                                 labels=tf.ones_like(self.d_logits_fake, name='g_ones')))
        with tf.name_scope('d_loss_fake'):
            self.d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_logits_fake,
                                                                                      labels=tf.zeros_like(self.d_logits_fake, name='d_zeros')))
        with tf.name_scope('d_loss'):
            self.d_loss = self.d_loss_real + self.d_loss_fake

        self.d_losses.append(self.d_loss)
        self.g_losses.append(self.g_loss)

        t_vars = tf.trainable_variables()
        self.d_vars = [var for var in t_vars if 'discriminator' in var.name]
        self.g_vars = [var for var in t_vars if 'generator' in var.name]

        self.saver = tf.train.Saver()

        self.d_summary_op = tf.summary.merge([tf.summary.scalar('d_loss_real', self.d_loss_real),
                                              tf.summary.scalar('d_loss', self.d_loss)])
        self.g_summary_op = tf.summary.merge([tf.summary.scalar('d_loss_fake', self.d_loss_fake),
                                              tf.summary.scalar('g_loss', self.g_loss)])

    def train(self, num_epochs):
        if self.verbose:
            print('starting training...')

        summary_writer = tf.summary.FileWriter(self.save_dir, graph=self.session.graph)
        d_train = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1)\
            .minimize(self.d_loss, var_list=self.d_vars)

        g_train = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1)\
            .minimize(self.g_loss, var_list=self.g_vars)

        tf.global_variables_initializer().run()

        if self.visualize:
            if not os.path.exists(self.samples_dir):
                os.makedirs(self.samples_dir)
            sample_z_placeholder = tf.placeholder(tf.float32, [None, self.z_dim], name='sample_z')
            sample_z = np.random.uniform(-1, 1, size=(self.num_sample, self.z_dim)).astype(np.float32)
            sampler = self.generator(sample_z_placeholder, is_training=True, reuse=True, sampling=True)

        sub_iter = 0
        start_time = time.time()
        for epoch in range(num_epochs):
            for b, batch in enumerate(self.batches):
                batch_z = np.random.uniform(-1, 1, [self.batch_size, self.z_dim]).astype(np.float32)

                _, summary = self.session.run([d_train, self.d_summary_op],
                                              feed_dict={self.inputs: batch, self.z: batch_z})
                summary_writer.add_summary(summary, sub_iter)

                _, summary = self.session.run([g_train, self.g_summary_op],
                                              feed_dict={self.z: batch_z})
                summary_writer.add_summary(summary, sub_iter)

                _, summary = self.session.run([g_train, self.g_summary_op],
                                              feed_dict={self.z: batch_z})
                summary_writer.add_summary(summary, sub_iter)

                d_error = self.d_loss.eval({self.inputs: batch, self.z: batch_z})
                g_error = self.g_loss.eval({self.z: batch_z})

                print("Epoch: [%4d/%4d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" % (epoch, num_epochs, b, len(self.batches),
                                                                                              time.time() - start_time,
                                                                                              d_error, g_error))
                sub_iter += 1

            if self.visualize and epoch % self.sample_epoch == 0:
                samples = self.session.run(sampler, feed_dict={sample_z_placeholder: sample_z})
                save_samples(samples, self.samples_dir, 'sample_epoch' + str(epoch) + '.jpg')

            if epoch == num_epochs - 1:
                self.saver.save(self.session, os.path.join(self.save_dir, "model.ckpt"),
                                global_step=epoch, write_meta_graph=True)
            elif self.save_on_epoch and epoch % num_epochs//self.save_epoch == 0:
                self.saver.save(self.session, os.path.join(self.save_dir, "model" + str(epoch) + ".ckpt"),
                                global_step=epoch, write_meta_graph=True)

    def discriminator(self, input, is_training=True, reuse=False):
        with tf.variable_scope('discriminator', reuse=reuse):
            w_conv1 = tf.get_variable(name='w_conv1',
                                      shape=[self.kernel_size, self.kernel_size,
                                             int(input.get_shape()[-1]), self.d_feature_dim],
                                      initializer=tf.truncated_normal_initializer(stddev=self.kernel_init_stddev))
            b_conv1 = tf.get_variable(name='b_conv1',
                                      shape=[self.d_feature_dim],
                                      initializer=tf.constant_initializer(self.bias_init))
            h_conv1 = conv_lrelu_layer(input=input,
                                       filter=w_conv1,
                                       bias=b_conv1,
                                       strides=[1, 2, 2, 1],
                                       padding='SAME',
                                       alpha=self.leaky_relu_alpha,
                                       do_batch_norm=False,
                                       is_training=is_training)

            w_conv2 = tf.get_variable(name='w_conv2',
                                      shape=[self.kernel_size, self.kernel_size,
                                             int(h_conv1.get_shape()[-1]), self.d_feature_dim*2],
                                      initializer=tf.truncated_normal_initializer(stddev=self.kernel_init_stddev))
            b_conv2 = tf.get_variable(name='b_conv2',
                                      shape=[self.d_feature_dim*2],
                                      initializer=tf.constant_initializer(self.bias_init))
            h_conv2 = conv_lrelu_layer(input=h_conv1,
                                       filter=w_conv2,
                                       bias=b_conv2,
                                       strides=[1, 2, 2, 1],
                                       padding='SAME',
                                       alpha=self.leaky_relu_alpha,
                                       do_batch_norm=True,
                                       momentum=self.momentum,
                                       epsilon=self.epsilon,
                                       is_training=is_training)

            w_conv3 = tf.get_variable(name='w_conv3',
                                      shape=[self.kernel_size, self.kernel_size,
                                             int(h_conv2.get_shape()[-1]), self.d_feature_dim*4],
                                      initializer=tf.truncated_normal_initializer(stddev=self.kernel_init_stddev))
            b_conv3 = tf.get_variable(name='b_conv3',
                                      shape=[self.d_feature_dim*4],
                                      initializer=tf.constant_initializer(self.bias_init))
            h_conv3 = conv_lrelu_layer(input=h_conv2,
                                       filter=w_conv3,
                                       bias=b_conv3,
                                       strides=[1, 2, 2, 1],
                                       padding='SAME',
                                       alpha=self.leaky_relu_alpha,
                                       do_batch_norm=True,
                                       momentum=self.momentum,
                                       epsilon=self.epsilon,
                                       is_training=is_training)

            w_conv4 = tf.get_variable(name='w_conv4',
                                      shape=[self.kernel_size, self.kernel_size,
                                             int(h_conv3.get_shape()[-1]), self.d_feature_dim*8],
                                      initializer=tf.truncated_normal_initializer(stddev=self.kernel_init_stddev))
            b_conv4 = tf.get_variable(name='b_conv4',
                                      shape=[self.d_feature_dim*8],
                                      initializer=tf.constant_initializer(self.bias_init))
            h_conv4 = conv_lrelu_layer(input=h_conv3,
                                       filter=w_conv4,
                                       bias=b_conv4,
                                       strides=[1, 2, 2, 1],
                                       padding='SAME',
                                       alpha=self.leaky_relu_alpha,
                                       do_batch_norm=True,
                                       momentum=self.momentum,
                                       epsilon=self.epsilon,
                                       is_training=is_training)

            h_conv4_shape = h_conv4.get_shape()
            w_linear = tf.get_variable(name='w_output',
                                       shape=[h_conv4_shape[1]*h_conv4_shape[2]*h_conv4_shape[3], 1],
                                       initializer=tf.random_normal_initializer(stddev=self.kernel_init_stddev))
            b_linear = tf.get_variable(name='b_output',
                                       shape=[1],
                                       initializer=tf.constant_initializer(self.bias_init))
            h_linear = tf.matmul(tf.reshape(h_conv4, [int(input.get_shape()[0]), -1]), w_linear) + b_linear

        return h_linear

    def generator(self, z, is_training=True, reuse=False, sampling=False):
        with tf.variable_scope('generator', reuse=reuse):
            if sampling:
                num_examples = self.num_sample
            else:
                num_examples = self.batch_size

            shape2 = conv_output_shape(self.output_shape, 2)
            shape4 = conv_output_shape(shape2, 2)
            shape8 = conv_output_shape(shape4, 2)
            shape16 = conv_output_shape(shape8, 2)

            w_linear = tf.get_variable(name='w_input',
                                       shape=[self.z_dim, 16*self.g_feature_dim*shape16[0]*shape16[1]],
                                       initializer=tf.random_normal_initializer(stddev=self.kernel_init_stddev))
            b_linear = tf.get_variable(name='b_input',
                                       shape=[1],
                                       initializer=tf.constant_initializer(self.bias_init))
            h_linear = tf.matmul(z, w_linear) + b_linear
            input_layer = tf.reshape(h_linear, [num_examples, shape16[0], shape16[1], 16*self.g_feature_dim])
            input_layer = tf.nn.relu(input_layer)

            w_conv1 = tf.get_variable(name='w_conv1',
                                      shape=[self.kernel_size, self.kernel_size,
                                             self.d_feature_dim*8, int(input_layer.get_shape()[-1])],
                                      initializer=tf.random_normal_initializer(stddev=self.kernel_init_stddev))
            b_conv1 = tf.get_variable(name='b_conv1',
                                      shape=[self.d_feature_dim*8],
                                      initializer=tf.constant_initializer(self.bias_init))
            h_conv1 = conv_relu_transpose_layer(input=input_layer,
                                                filter=w_conv1,
                                                bias=b_conv1,
                                                output_shape=[num_examples, shape8[0], shape8[1], self.d_feature_dim*8],
                                                strides=[1, 2, 2, 1],
                                                padding='SAME',
                                                do_batch_norm=True,
                                                is_training=is_training)

            w_conv2 = tf.get_variable(name='w_conv2',
                                      shape=[self.kernel_size, self.kernel_size,
                                             self.d_feature_dim*4, int(h_conv1.get_shape()[-1])],
                                      initializer=tf.random_normal_initializer(stddev=self.kernel_init_stddev))
            b_conv2 = tf.get_variable(name='b_conv2',
                                      shape=[self.d_feature_dim*4],
                                      initializer=tf.constant_initializer(self.bias_init))
            h_conv2 = conv_relu_transpose_layer(input=h_conv1,
                                                filter=w_conv2,
                                                bias=b_conv2,
                                                output_shape=[num_examples, shape4[0], shape4[1], self.d_feature_dim*4],
                                                strides=[1, 2, 2, 1],
                                                padding='SAME',
                                                do_batch_norm=True,
                                                is_training=is_training)

            w_conv3 = tf.get_variable(name='w_conv3',
                                      shape=[self.kernel_size, self.kernel_size,
                                             self.d_feature_dim*2, int(h_conv2.get_shape()[-1])],
                                      initializer=tf.random_normal_initializer(stddev=self.kernel_init_stddev))
            b_conv3 = tf.get_variable(name='b_conv3',
                                      shape=[self.d_feature_dim*2],
                                      initializer=tf.constant_initializer(self.bias_init))
            h_conv3 = conv_relu_transpose_layer(input=h_conv2,
                                                filter=w_conv3,
                                                bias=b_conv3,
                                                output_shape=[num_examples, shape2[0], shape2[1], self.d_feature_dim*2],
                                                strides=[1, 2, 2, 1],
                                                padding='SAME',
                                                do_batch_norm=True,
                                                is_training=is_training)

            w_conv4 = tf.get_variable(name='w_conv4',
                                      shape=[self.kernel_size, self.kernel_size,
                                             self.output_shape[2], int(h_conv3.get_shape()[-1])],
                                      initializer=tf.random_normal_initializer(stddev=self.kernel_init_stddev))
            b_conv4 = tf.get_variable(name='b_conv4',
                                      shape=[self.output_shape[2]],
                                      initializer=tf.constant_initializer(self.bias_init))
            h_conv4 = conv_relu_transpose_layer(input=h_conv3,
                                                filter=w_conv4,
                                                bias=b_conv4,
                                                output_shape=[num_examples] + self.output_shape,
                                                strides=[1, 2, 2, 1],
                                                padding='SAME',
                                                do_relu=False,
                                                do_batch_norm=True,
                                                is_training=is_training)
            return tf.nn.tanh(h_conv4)

    def test(self, num_samples, checkpoint_dir, checkpoint_name, samples_dir):
        self.saver.restore(self.session, os.path.join(checkpoint_dir, checkpoint_name))
        sample_z_placeholder = tf.placeholder(tf.float32, [None, self.z_dim], name='sample_z')
        # np.random.seed(1234)
        sample_z = np.random.uniform(-1, 1, size=(self.num_sample, self.z_dim)).astype(np.float32)
        # sample_z = .5*np.ones((self.num_sample, self.z_dim))
        # sample_z[:, 2:70] = 0
        sampler = self.generator(sample_z_placeholder, is_training=True, reuse=True, sampling=True)
        samples = []
        while len(samples) < math.ceil(num_samples/64.):
            samples.append(self.session.run(sampler, feed_dict={sample_z_placeholder: sample_z}))

        samples = np.concatenate(samples, axis=0)
        save_samples(samples[:num_samples], samples_dir, 'sample.jpg')

    def generate_individual_samples(self, num_samples, checkpoint_dir, checkpoint_name, samples_dir):
        self.saver.restore(self.session, os.path.join(checkpoint_dir, checkpoint_name))
        sample_z_placeholder = tf.placeholder(tf.float32, [None, self.z_dim], name='sample_z')
        sample_z = np.random.uniform(-1, 1, size=(num_samples, self.z_dim)).astype(np.float32)
        sampler = self.generator(sample_z_placeholder, is_training=True, reuse=True, sampling=True)
        z_batches = list(make_batches(sample_z, self.num_sample))
        samples = []
        for z_batch in z_batches:
            if z_batch.shape[0] == self.num_sample:
                samples.append(self.session.run(sampler, feed_dict={sample_z_placeholder: z_batch}))

        samples = np.concatenate(samples, axis=0)
        save_individual_samples(samples[:num_samples], samples_dir, 'sample.jpg')
        pass

    def create_turing_test(self, num_samples, checkpoint_dir, checkpoint_name, fakes_dir, reals_dir):
        # self.saver.restore(self.session, os.path.join(checkpoint_dir, checkpoint_name))
        # sample_z_placeholder = tf.placeholder(tf.float32, [None, self.z_dim], name='sample_z')
        # sample_z = np.random.uniform(-1, 1, size=(num_samples, self.z_dim)).astype(np.float32)
        # sampler = self.generator(sample_z_placeholder, is_training=True, reuse=True, sampling=True)
        # z_batches = list(make_batches(sample_z, self.num_sample))
        # samples = []
        # for z_batch in z_batches:
        #     if z_batch.shape[0] == self.num_sample:
        #         samples.append(self.session.run(sampler, feed_dict={sample_z_placeholder: z_batch}))
        #
        # samples = np.concatenate(samples, axis=0)
        # save_individual_samples(gaussian_filter(samples[:num_samples], sigma=.3), fakes_dir, 'fake.jpg')
        # # noise = 0.05 * np.random.rand(num_samples, self.train_data.shape[1], self.train_data.shape[2], self.train_data.shape[3]) - 0.05
        real_inds = np.random.randint(0, self.train_data.shape[0], (num_samples,))
        # # noised_train = self.train_data[real_inds] + noise
        # # noised_train = np.clip(noised_train, 0, 1)
        blurred_train = gaussian_filter(self.train_data[real_inds], sigma=.4)
        save_individual_samples(blurred_train, reals_dir, 'real.jpg')
        pass


def main():
    data_path = sys.argv[1]
    num_epochs = 1000

    data = np.load(data_path)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.InteractiveSession()
    dcgan = DCGAN(session=session,
                  data=data[:1],
                  batch_size=256,
                  input_shape=[176, 176, 3],
                  output_shape=[80, 80, 3])

    # dcgan.train(num_epochs)
    fakes_dir = './male_generated/'
    if not os.path.exists(fakes_dir):
        os.mkdir(fakes_dir)
    #
    # reals_dir = './reals/'
    # if not os.path.exists(reals_dir):
    #     os.mkdir(reals_dir)


    # dcgan.create_turing_test(1000, './logs/run_at_80x80_female_500/', 'model.ckpt-499', fakes_dir, reals_dir)
    # dcgan.create_turing_test(100, './logs/overnight_270_women1/', 'model49.ckpt-49', fakes_dir, reals_dir)
    # dcgan.generate_individual_samples(10000, './logs/run_at_80x80_female_500/', 'model.ckpt-499', fakes_dir)
    dcgan.generate_individual_samples(5000, './logs/run_at_80x80_male_500/', 'model.ckpt-499', fakes_dir)

    session.close()


if __name__ == "__main__":
    main()
    pass
