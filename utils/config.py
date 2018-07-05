# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import json


def save_config(args_dict, save_path):
    """
    Save model's config

    :param args_dict: config dictionary
    :param save_path: path for save
    :return:
    """
    with open(save_path, 'w') as fd:
        json.dump(args_dict, fd)


def load_config(path):
    """
    load model's config

    :param path: path for load
    :return: config
    """
    with open(path, 'r') as fd:
        return json.load(fd)