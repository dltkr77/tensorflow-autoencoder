# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import numpy as np


def next_mnist_data(mnist, attr, batch_size=128):
    """
    get next batch with mnist

    :param mnist: mnist dataset
    :param attr: 'train' or 'test'
    :param batch_size: batch size
    :return: generator (x_list, y_list)
    """
    data = mnist.__getattribute__(attr).images
    labels = np.asarray(mnist.__getattribute__(attr).labels, dtype=np.int32)

    xs = []
    ys = []
    for x, y in zip(data, labels):
        xs.append(x)
        ys.append(y)
        if len(xs) >= batch_size:
            yield xs, ys
            xs = []
            ys = []

    yield xs, ys