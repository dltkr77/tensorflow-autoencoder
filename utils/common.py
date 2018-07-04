# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import tensorflow as tf


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