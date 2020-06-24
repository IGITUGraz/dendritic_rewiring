# -*- coding: utf-8 -*-
"""TODO"""

import collections
import sys

import core.core_global as core
from core.core_definitions import simulation_timestep
from core.monitor import Monitor


class VoltageMonitor(Monitor):
    """TODO"""

    def __init__(self, source, uid, filename, stepsize=simulation_timestep,
                 paste_spikes=False, paste_spikes_from=None):
        """TODO"""
        super().__init__(filename)

        self.source = source
        self.uid = uid
        self.stop_time = sys.maxsize
        self.stepsize = int(stepsize / simulation_timestep)
        self.paste_spikes = paste_spikes
        self.paste_spikes_from = paste_spikes_from
        self.pasted_spike_height = -25

        if self.stepsize < 1:
            self.stepsize = 1

        if isinstance(self.uid, collections.Iterable):
            self.gid = source.rank2global(self.uid[1])
        else:
            self.gid = source.rank2global(self.uid)

        if isinstance(self.uid, collections.Iterable):
            if self.uid[1] < self.source.get_post_size():
                core.kernel.register_device(self)
                self.outfile.write("# Recording from compartment"
                                   " %d of neuron %d\n".encode()
                                   % (self.uid[0], self.gid))
        else:
            if self.uid < self.source.get_post_size():
                core.kernel.register_device(self)
                self.outfile.write("# Recording from neuron %d\n".encode()
                                   % (self.gid))

    def execute(self):
        """TODO"""
        if not self.active:
            return

        if core.kernel.get_clock() < self.stop_time:
            if self.paste_spikes:
                for spike in self.paste_spikes_from.get_spikes_immediate():
                    if spike == self.uid:
                        voltage = self.pasted_spike_height
                        self.outfile.write("%.6f %.6f\n".encode()
                                           % (core.kernel.get_time(),
                                              voltage))
                pass
            if core.kernel.get_clock() % self.stepsize == 0:
                self.outfile.write("%.6f %.6f\n".encode()
                                   % (core.kernel.get_time(),
                                      self.source.mem[self.uid]))

    def record_for(self, duration=10.0):
        """TODO"""
        self._set_stop_time(duration)

    def _set_stop_time(self, duration):
        """TODO"""
        if duration < 0:
            core.logger.warning("Warning: Negative stop times not"
                                " supported -- ingoring.")
        else:
            self.stop_time = (core.kernel.get_clock() +
                              duration / simulation_timestep)
