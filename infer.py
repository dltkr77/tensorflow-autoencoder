# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import argparse
import os

import tensorflow as tf

from utils import restore
from utils import get_network
from utils import build_graph
from utils import next_mnist_data


g_logger = tf.logging
g_logger.set_verbosity(tf.logging.DEBUG)


def main(args):
    # create autoencoder
    ae = get_network(args.hiddens, logger=g_logger)

    # build graph
    sess, saver = build_graph(ae, input_shape=[None, 784])
    restore(sess, saver, args.restore)

    # make result directory if not exists
    if not os.path.exists(args.result):
        os.makedirs(args.result)

    # use mnist for test
    mnist = tf.contrib.learn.datasets.load_dataset('mnist')

    for x, y in next_mnist_data(mnist, 'test'):
        print(x, y)
        z = sess.run(ae.z, feed_dict={ae.x: x})
        print(z)
        exit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('restore', type=str, help='the directory for restore')
    parser.add_argument('--hiddens', type=str, default="4,2", help='comma separated hidden dimensions')
    parser.add_argument('--result', type=str, default='./results', help='the directory of result')
    args = parser.parse_args()
    main(args)