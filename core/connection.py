# -*- coding: utf-8 -*-
"""TODO"""

import numpy as np


class Connection:
    """TODO"""

    def __init__(self, source, destination, transmitter, name="Connection"):
        """TODO"""
        self.source = source
        self.destination = destination
        self.transmitter = transmitter
        self.name = name

        self.set_target(self.transmitter)

    def get_destination(self):
        """TODO"""
        return self.destination

    def get_source(self):
        """TODO"""
        return self.source

    def get_nonzero(self):
        """TODO"""
        return np.count_nonzero(self.w)

    def get_pre_trace(self, tau):
        """TODO"""
        return self.source.get_pre_trace(tau)

    def get_post_trace(self, tau):
        """TODO"""
        return self.destination.get_post_trace(tau)

    def get_pre_spikes(self):
        """TODO"""
        return self.source.get_spikes()

    def get_post_spikes(self):
        """TODO"""
        return self.destination.get_spikes_immediate()

    def set_target(self, target):
        """TODO"""
        self.target_state_vector = target

    def transmit(self, gid, amount):
        """TODO"""
        self.targeted_transmit(self.destination, self.target_state_vector, gid,
                               amount)

    def targeted_transmit(self, destination, target_state_vector, gid,
                          amount):
        """TODO"""
        pass

    def evolve(self):
        """TODO"""
        pass
