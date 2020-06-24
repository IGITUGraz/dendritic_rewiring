# -*- coding: utf-8 -*-
"""TODO"""

import numpy as np

import core.core_global as core
from core.core_definitions import simulation_timestep
from core.spiking_group import SpikingGroup


class PoissonGroup(SpikingGroup):
    """TODO"""

    def __init__(self, size, rate=1.0):
        """TODO"""
        super().__init__(size)

        core.kernel.register_spiking_group(self)

        if self.evolve_locally:
            self.set_rate(rate)

            self.rng = np.random.RandomState(seed=core.kernel.get_seed())

    def evolve(self):
        """TODO"""
        spikes = np.flatnonzero(self.rng.uniform(
            size=self.rank_size) < self.lambd * simulation_timestep)
        for spike in spikes:
            self.push_spike(spike)

    def set_rate(self, rate):
        """TODO"""
        self.lambd = 1.0/(1.0/rate - simulation_timestep)

    def get_rate(self):
        """TODO"""
        return self.lambd
