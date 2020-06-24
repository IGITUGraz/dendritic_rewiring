# -*- coding: utf-8 -*-
"""TODO"""

from core.spiking_group import NodeDistributionMode, SpikingGroup


class McNeuronGroup(SpikingGroup):
    """TODO"""

    def __init__(self, n, m, mode=NodeDistributionMode.AUTO):
        """TODO"""
        super().__init__(n, mode)

        if self.evolve_locally:
            self.num_compartments = m
            self.group_name = "McNeuronGroup"

    def evolve(self):
        pass
