# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import tensorflow as tf

from model import AutoEncoder


def restore(sess, saver, restore_dir):
    """
    Restore model

    :param sess: tensorflow session
    :param saver: tensorflow saver
    :param restore_dir: the directory for restore
    :return:
    """
    checkpoint = tf.train.get_checkpoint_state(restore_dir)
    if checkpoint is None:
        raise FileNotFoundError('not found model: {}'.format(restore_dir))

    saver.restore(sess, checkpoint.model_checkpoint_path)


def get_network(args_hiddens, logger=None):
    """
    get autoencoder

    :param args_hiddens: args.hiddens (comma separated)
    :param logger: logger
    :return: autoencoder model
    """
    hidden_dims = [int(h) for h in args_hiddens.split(',')]
    return AutoEncoder(hidden_dims, logger=logger)


def build_graph(network, input_shape, dtype=tf.float64):
    """
    build graph

    :param network: network(model)
    :param input_shape: input data's shape
    :param dtype: type of input data
    :return: sess, saver
    """
    with tf.Graph().as_default():
        x_ph = tf.placeholder(dtype=dtype, shape=input_shape)
        network.build_graph(x_ph)
        sess = tf.Session()
        saver = tf.train.Saver()

    return sess, saver