# -*- coding: utf-8 -*-

"""TODO"""


class SpikeDelay:
    """TODO"""

    def __init__(self, delay):
        """TODO"""
        self.delay = delay
        self._clock = None
        self._delay_buffer = [[]] * self.delay

    def set_delay(self, delay):
        """TODO"""
        if delay == self.delay:
            return

        del self._delay_buffer
        self.delay = delay
        self._delay_buffer = [[]] * self.delay

    def get_spikes(self, position=1):
        """TODO"""
        return self._delay_buffer[(self._clock + position) % self.delay]

    def get_spikes_immediate(self):
        """TODO"""
        return self.get_spikes(0)

    def set_clock(self, clock):
        """TODO"""
        self._clock = clock
