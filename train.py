# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import argparse
import os

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from model import AutoEncoder
from utils import restore


g_logger = tf.logging
g_logger.set_verbosity(tf.logging.DEBUG)


def next_batch(mnist, batch_size=128):
    """
    get next batch with mnist

    :param mnist: mnist dataset
    :param batch_size: batch size
    :return: generator (x_list, y_list)
    """
    train_data = mnist.train.images
    train_labels = np.asarray(mnist.train.labels, dtype=np.int32)

    xs = []
    ys = []
    for x, y in zip(train_data, train_labels):
        xs.append(x)
        ys.append(y)
        if len(xs) >= batch_size:
            yield xs, ys
            xs = []
            ys = []

    yield xs, ys


def scatter(scatter_data, result_dir, f_name):
    """
    Scatter and save figure

    :param scatter_data: scattner_data(dict)
    :param result_dir: the directory for save images
    :param f_name: file name for save
    :return:
    """
    plt.title('Dimension-Reduction Visualization')
    for key, value in scatter_data.items():
        plt.scatter(value['x'], value['y'], marker=(1, 2), label=str(key))

    plt.legend(loc='best')
    plt.savefig(os.path.join(result_dir, '{}.png'.format(f_name)))


def main(args):
    # create autoencoder
    ae = AutoEncoder([4, 2], lr=args.lr, logger=g_logger)

    # build graph
    with tf.Graph().as_default():
        x_ph = tf.placeholder(dtype=tf.float64, shape=[None, 784])
        ae.build_graph(x_ph)
        sess = tf.Session()
        saver = tf.train.Saver()

        if args.restore:
            restore(sess, saver, args.restore)
        else:
            g_logger.info('Initialize the model')
            sess.run(tf.global_variables_initializer())

    # make result directory if not exists
    if not os.path.exists(args.result):
        os.makedirs(args.result)

    # use mnist for test
    mnist = tf.contrib.learn.datasets.load_dataset('mnist')

    figure = plt.figure(figsize=(8, 8))
    scatter_data = {}

    last_epoch = 0
    try:
        # Learn number of epoch times
        nodes = [ae.train, ae.loss, ae.z, ae.x_]
        for i in range(1, args.epoch+1):
            losses = 0
            cnt = 0
            # get data with batch size
            for x, y in next_batch(mnist):
                _, loss, z, x_ = sess.run(nodes, feed_dict={ae.x: x})
                # make scatter data with latent variables(z)
                for key, value in zip(y, z):
                    if key not in scatter_data:
                        scatter_data[key] = {'x': [], 'y': []}

                    scatter_data[key]['x'].append(value[0])
                    scatter_data[key]['y'].append(value[1])

                losses += loss
                cnt += 1
                last_epoch += 1

            g_logger.info('epoch: {}, loss: {}'.format(i, losses/cnt))
            scatter(scatter_data, args.result, i)
            figure.clear()
            scatter_data.clear()

        # save checkpoint
        saver.save(sess, args.result + '/checkpoint', global_step=args.epoch)
    except KeyboardInterrupt:
        saver.save(sess, args.result + '/checkpoint', global_step=last_epoch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=1e-04, help='learning rate')
    parser.add_argument('--restore', type=str, default=None, help='the directory for restore')
    parser.add_argument('--result', type=str, default='./results', help='the directory of result')
    parser.add_argument('--epoch', type=int, default=1000, help='number of epoch')
    args = parser.parse_args()
    main(args)