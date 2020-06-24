# -*- coding: utf-8 -*-
"""TODO"""

import numpy as np


class Trace2D:
    """TODO"""

    def __init__(self, size):
        """TODO"""
        self.size = size
        self.val = np.zeros(size)

    def inc(self, i, j):
        """TODO"""
        self.val[i, j] += 1

    def set(self, i, j):
        """TODO"""
        self.val[i, j] = 1
