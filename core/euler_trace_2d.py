# -*- coding: utf-8 -*-
"""TODO"""

import numpy as np

from core.core_definitions import simulation_timestep
from core.trace_2d import Trace2D


class EulerTrace2D(Trace2D):
    """TODO"""

    def __init__(self, size, tau):
        """TODO"""
        super().__init__(size)

        self.tau = tau

        self.calculate_scale_constants()

    def calculate_scale_constants(self):
        """TODO"""
        self.scale_trace = np.exp(-simulation_timestep / self.tau)

    def evolve(self):
        """TODO"""
        np.multiply(self.val, self.scale_trace, self.val)

    def get_tau(self):
        """TODO"""
        return self.tau
