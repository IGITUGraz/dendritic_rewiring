# -*- coding: utf-8 -*-
"""TODO"""

import numpy as np


class Trace:
    """TODO"""

    def __init__(self, size):
        """TODO"""
        self.size = size
        self.val = np.zeros(size)

    def inc(self, i):
        """TODO"""
        self.val[i] += 1

    def set(self, i):
        """TODO"""
        self.val[i] = 1
