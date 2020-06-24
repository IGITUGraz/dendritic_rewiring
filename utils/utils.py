# /usr/bin/env python
"""This script contains some handy functions."""

import ruamel_yaml as yaml

import numpy as np


def load_configuration(filepath):
    with open(filepath, "r") as f:
        config = yaml.safe_load(f)

    return config


def write_configuration(filepath, config):
    with open(filepath, "w") as f:
        config = yaml.dump(config, f, default_flow_style=False)

    return config


# TODO deprecated?
def reject_outliers(data, m=2):
    r = data.copy()
    r[abs(data - np.mean(data)) < m * np.std(data)] = 0

    return r
