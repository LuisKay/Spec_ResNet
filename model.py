# -*- coding: UTF-8 -*-
# tensorflow version 1.0
import tensorflow as tf
from tensorflow.python.training import moving_averages
from tensorflow.python.ops import control_flow_ops
import six
import numpy as np



class model(object):
    def __init__(self,hps,data,labels):
        self.hps = hps
        self.data = data
        self.labels = labels


    def build_model(self):
        self.data = tf.cast(self.data, tf.float32)
        spec_res = tf.reshape(self.data, [-1, self.hps.feature_row, self.hps.feature_col, 1])
        spec_filtered = self.fixed_conv('fixed_conv_filter', spec_res, 3, 1, self.hps.channels, [1, 1, 1, 1])

        with tf.variable_scope('init'):
            x = self._conv('init_conv', spec_filtered, 3, self.hps.channels, 10, self._stride_arr(1))

        strides = [1, 2, 2]
        activate_before_residual = [True, False, False]
        res_func = self._residual
        filters = [10, 10, 20, 40]
        with tf.variable_scope('unit_1_0'):
            x = res_func(x, filters[0], filters[1], self._stride_arr(strides[0]),
                         activate_before_residual[0])
            for i in six.moves.range(1, self.hps.num_residual_units):
                with tf.variable_scope('unit_1_%d' % i):
                    x = res_func(x, filters[1], filters[1], self._stride_arr(1), False)

        with tf.variable_scope('unit_2_0'):
            x = res_func(x, filters[1], filters[2], self._stride_arr(strides[1]),
                         activate_before_residual[1])
            for i in six.moves.range(1, self.hps.num_residual_units):
                with tf.variable_scope('unit_2_%d' % i):
                    x = res_func(x, filters[2], filters[2], self._stride_arr(1), False)

        with tf.variable_scope('unit_3_0'):
            x = res_func(x, filters[2], filters[3], self._stride_arr(strides[2]),
                         activate_before_residual[2])
            for i in six.moves.range(1, self.hps.num_residual_units):
                with tf.variable_scope('unit_3_%d' % i):
                    x = res_func(x, filters[3], filters[3], self._stride_arr(1), False)

        with tf.variable_scope('unit_last'):
            x = self._batch_norm('final_bn', x)
            x = self._relu(x)
            x = self._global_avg_pool(x)

        with tf.variable_scope('logit'):
            logits = self._fully_connected(x, self.hps.num_classes)


        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                       labels=tf.reshape(self.labels, [self.hps.batch_size]),
                                                                       name='cross_entropy_per_example')
        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
        decay = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))		
        self.cost = cross_entropy_mean + decay
		self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.learning_rate = tf.train.exponential_decay(self.hps.start_lrn_rate, self.global_step,
                                                   100, self.hps.decay_rate, staircase=True, name='lr')

        if self.hps.optimizer == 'sgd':
            self.train_op = tf.train.GradientDescentOptimizer(self.hps.start_lrn_rate).minimize(self.cost, global_step=self.global_step)
        elif self.hps.optimizer == 'adam':
            self.train_op = tf.train.AdamOptimizer(self.hps.start_lrn_rate).minimize(self.cost, global_step=self.global_step)
        
		prediction = tf.cast(tf.reshape(tf.argmax(tf.nn.softmax(logits), axis=1), [self.hps.batch_size, 1]), tf.int32)
        correct = tf.equal(prediction, self.labels)
        truth = tf.equal(self.labels, 1)

        true_positive = tf.logical_and(truth, correct)
        true_negative = tf.logical_and(tf.logical_not(truth), correct)

        self.correct_count = tf.reduce_sum(tf.cast(correct, tf.int32), name='acc')
        self.tp_count = tf.reduce_sum(tf.cast(true_positive, tf.int32), name='tp')
        self.tn_count = tf.reduce_sum(tf.cast(true_negative, tf.int32), name='tn')

    def _residual(self, x, in_filter, out_filter, stride,
                  activate_before_residual=False):
        """Residual unit with 2 sub layers."""
        if activate_before_residual:
            with tf.variable_scope('shared_activation'):
                x = self._batch_norm('init_bn', x)
                x = self._activate_f(x)
                orig_x = x
        else:
            with tf.variable_scope('residual_only_activation'):
                orig_x = x
                x = self._batch_norm('init_bn', x)
                x = self._activate_f(x)

        with tf.variable_scope('sub1'):
            x = self._conv('conv1', x, 3, in_filter, out_filter, stride)

        with tf.variable_scope('sub2'):
            x = self._batch_norm('bn2', x)
            x = self._activate_f(x)
            x = self._conv('conv2', x, 3, out_filter, out_filter, [1, 1, 1, 1])

        with tf.variable_scope('sub_add'):
            if in_filter != out_filter:
                orig_x = tf.nn.avg_pool(orig_x, stride, stride, 'VALID')
                orig_x = tf.pad(orig_x, [[0, 0], [0, 0], [0, 0],
                                         [(out_filter - in_filter) // 2,
										 (out_filter - in_filter) // 2]])
            x += orig_x

        tf.logging.debug('image after unit %s', x.get_shape())
        return x

    # define the strides on convolutional operations
    def _stride_arr(self,stride):
        """Map a stride scalar to the stride array for tf.nn.conv2d."""
        return [1, stride, stride, 1]

    # convolutional operation with regularizer
    def _conv(self, name, x, filter_size, in_filters, out_filters, strides):
		regularizer = tf.contrib.layers.l2_regularizer(scale=self.hps.weight_decay_rate)
        with tf.variable_scope(name):
            n = filter_size * filter_size * out_filters
            kernel = tf.get_variable(
                'DW', [filter_size, filter_size, in_filters, out_filters], tf.float32,
				initializer = tf.truncated_normal_initializer(stddev=0.1),
				regularizer = regularizer)
            return tf.nn.conv2d(x, kernel, strides, padding='SAME')

#    def _decay(self):
#        """L2 weight decay loss."""
#        costs = []
#        for var in tf.trainable_variables():
#            if var.op.name.find(r'DW') > 0:
#                costs.append(tf.nn.l2_loss(var))
#
#        return tf.multiply(self.hps.weight_decay_rate, tf.add_n(costs))

    # activation function
    def _activate_f(self, x):
        # return tf.where(tf.less(x, 0.0), leakiness * x, x, name='leaky_relu')
        return tf.nn.relu(x, name='relu')

    def _fully_connected(self, x, out_dim):
        """FullyConnected layer for final output with regularizer."""
		regularizer = tf.contrib.layers.l2_regularizer(scale=self.hps.weight_decay_rate)
        x = tf.reshape(x, [self.hps.batch_size, -1])
        w = tf.get_variable(
            'DW', [x.get_shape()[1], out_dim],
            initializer=tf.truncated_normal_initializer(stddev=0.5))
        b = tf.get_variable('biases', [out_dim], dtype = tf.float32,
                            initializer = tf.constant_initializer(0, tf.float32),
							regularizer = regularizer)
        return tf.nn.xw_plus_b(x, w, b)

    def _batch_norm(self, name, x):
        """Batch normalization."""
        with tf.variable_scope(name):
            params_shape = [x.get_shape()[-1]]
            #return a tuple
            beta = tf.get_variable(
                'beta', params_shape, tf.float32,
                initializer=tf.constant_initializer(0.0, tf.float32))
            gamma = tf.get_variable(
                'gamma', params_shape, tf.float32,
                initializer=tf.constant_initializer(1.0, tf.float32))
            mean, variance = tf.nn.moments(x, [0, 1, 2], name='moments')
            moving_mean = tf.get_variable(
                'moving_mean', params_shape, tf.float32,
                initializer=tf.constant_initializer(0.0, tf.float32),
                trainable=False)
            moving_variance = tf.get_variable(
                'moving_variance', params_shape, tf.float32,
                initializer=tf.constant_initializer(1.0, tf.float32),
                trainable=False)
            updated_mean = moving_averages.assign_moving_average(moving_mean,
                                                                mean, self.hps.BN_decay)
            updated_variance = moving_averages.assign_moving_average(moving_variance,
                                                                    variance, self.hps.BN_decay)

            mean, variance = control_flow_ops.cond(tf.cast(self.hps.is_training,tf.bool),lambda:(mean,variance),
                                                   lambda:(updated_mean,updated_variance))
            # epsilon used to be 1e-5. Maybe 0.001 solves NaN problem in deeper net.
            y = tf.nn.batch_normalization(
                x, mean, variance, beta, gamma, 1e-5)
            y.set_shape(x.get_shape())
            return y

    def _global_avg_pool(self, x):
        """global average pooling layer"""
        assert x.get_shape().ndims == 4
        return tf.reduce_mean(x, [1, 2], name='feature')

    def fixed_conv(self, name, x, filter_size, in_filters, out_filters, strides):
        """
			Convolutional layer with parameters fixed. You can preprocess the spectrogram matrix with parameters-fixed filtes.
			Args:
				name: op name.
				x: input tensor, the shape of which is 'batch_size * feature_row * feature_col * in_filters'.
				filter_size: the size for convolutional filters. 3 or 5.
				in_filters: the channels of input tensor.
				out_filters: the channels of output tenso.
				strides: step of filter moing along the feature matrix.
			return: A tensor, the shape of which is 'batch_size * feature_row * feature_col * out_filters'.
		"""
        with tf.variable_scope(name):
            kernel = tf.get_variable(
                name, [filter_size, filter_size, in_filters, out_filters],
                tf.float32, initializer=tf.constant_initializer([
            #       0, 0, 0, 0, -1, 1, 0, 0, 0]))
                    0, -1, 0, 0, 1, 0, 0, 0, 0,
                    0, 1, 0, 0, -2, 0, 0, 1, 0,
                    0, 0, 0, -1, 1, 0, 0, 0, 0,
                    0, 0, 0, 1, -2, 1, 0, 0, 0]), trainable=False)
            #       0, 1, 0, 1, -2, 1, 0, 1, 0,
            #       1, 1, 1, 1, -2, 1, 1, 1, 1,
            #       0, 0, 1, 1, -2, 1, 1, 0, 0,
            #       0, 1, 1, 1, -2, 1, 1, 1, 0,
            #       -1, 2, -1, 2, -4, 2, 0, 0, 0,
            #       -1, 2, -1, 2, -4, 2, -1, 2, -1]))
            return tf.nn.conv2d(x, kernel, strides, padding='SAME')
