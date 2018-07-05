# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import argparse
import os

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from utils import restore
from utils import get_network
from utils import build_graph
from utils import next_mnist_data


g_logger = tf.logging
g_logger.set_verbosity(tf.logging.DEBUG)


def save_mnist_images(data, result_dir, f_name, suffix='', row_col_size=1):
    """
    Save MNIST images

    :param data: data
    :param result_dir: the directory for save images
    :param f_name: file name for save
    :param prefix: the prefix of file name
    :return:
    """
    fig, axes = plt.subplots(row_col_size, row_col_size,
                             figsize=(row_col_size, row_col_size))
    for i in range(row_col_size):
        for j in range(row_col_size):
            axes[i][j].set_axis_off()
            axes[i][j].imshow(np.reshape(data[i*10 + j], (28, 28)))

    img_path = os.path.join(result_dir, '{}_{}.png'.format(f_name, suffix))
    plt.savefig(img_path)
    g_logger.info('save image: {}'.format(img_path))
    plt.close(fig)


def main(args):
    # create autoencoder
    ae = get_network(args.hiddens, logger=g_logger)

    # build graph
    sess, saver = build_graph(ae, input_shape=[None, 784])
    restore(sess, saver, args.restore)

    test_result = os.path.join(args.result, 'test')
    # make result directory if not exists
    if not os.path.exists(test_result):
        os.makedirs(test_result)

    # use mnist for test
    mnist = tf.contrib.learn.datasets.load_dataset('mnist')

    row_col_size = 10
    cnt = 0
    for x, y in next_mnist_data(mnist, 'test', batch_size=row_col_size**2):
        x_ = sess.run(ae.x_, feed_dict={ae.x: x})
        save_mnist_images(x, test_result, cnt, suffix='original', row_col_size=row_col_size)
        save_mnist_images(x_, test_result, cnt, suffix='reconstruct', row_col_size=row_col_size)
        cnt += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('restore', type=str, help='the directory for restore')
    parser.add_argument('--hiddens', type=str, default="4,2", help='comma separated hidden dimensions')
    parser.add_argument('--result', type=str, default='./results', help='the directory of result')
    args = parser.parse_args()
    main(args)