# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import logging

import tensorflow as tf


class AutoEncoder:
    """
    Autoencoder implemented with tensorflow
    """
    def __init__(self, encoder_dims, decoder_dims=None, lr=1e-05, logger=None):
        """
        Constructor of the model

        :param encoder_dims:
        encoder's dimensions list.
        ex) [4, 2] => x -> 4dim -> 2dim(z)

        :param decoder_dims:
        decoder's dimensions list.
        ex) None, encoder_dims: [4, 2] => z -> 4dim
        ex) [2, 4]: z -> 2dim -> 4dim

        :param lr: learning rate
        """
        self.encoder_dims = encoder_dims
        if decoder_dims:
            self.decoder_dims = decoder_dims
        else:
            self.decoder_dims = list(reversed(encoder_dims))[1:]
        self.learning_rate = lr

        if not logger:
            self.logger = logging.getLogger(__name__)
            self.logger.setLevel('DEBUG')
        else:
            self.logger = logger

    def build_graph(self, x_ph):
        """
        Build graph

        :param x_ph: the placeholder of input
        :return: None
        """
        with tf.variable_scope('encoder'):
            self.x = x_ph
            self.logger.debug('x: {}'.format(self.x))

            self.z = self.encoder(self.x)
            self.logger.debug('z: {}'.format(self.z))

        with tf.variable_scope('decoder'):
            self.x_ = self.decoder(self.z)
            self.logger.debug('x_: {}'.format(self.x_))

        with tf.name_scope('train'):
            self.loss = tf.losses.mean_squared_error(self.x, self.x_)
            self.logger.debug('loss: {}'.format(self.loss))

            self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
            self.logger.debug('optimizer: {}'.format(self.optimizer))

            self.train = self.optimizer.minimize(self.loss)
            self.logger.debug('train: {}'.format(self.train))

    def encoder(self, x):
        """
        Encoder
        activation: leaky_relu for test

        :param x: input
        :return: encoded_x
        """
        result_x = x
        for dim in self.encoder_dims:
            result_x = tf.layers.dense(result_x, dim, activation=tf.nn.leaky_relu)
            self.logger.debug('result_x: {}'.format(result_x))

        return result_x

    def decoder(self, z):
        """
        Decoder
        activation: leaky_relu for test

        :param z: latent variables
        :return: decoded(x_hat)
        """
        result_z = z
        for dim in self.decoder_dims + [self.x.shape[-1]]:
            result_z = tf.layers.dense(result_z, dim, activation=tf.nn.leaky_relu)
            self.logger.debug('result_z: {}'.format(result_z))

        return result_z