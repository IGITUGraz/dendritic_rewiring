# -*- coding: utf-8 -*-
"""TODO"""

import core.core_global as core
from core.monitor import Monitor


class SpikeMonitor(Monitor):
    """TODO"""

    def __init__(self, source, filename, from_gid=0, to_gid=None, every=1):
        """TODO"""
        super().__init__(filename)

        core.kernel.register_device(self)

        self.source = source
        self.from_gid = from_gid
        self.to_gid = to_gid
        self.every = every
        if self.to_gid is None:
            self.to_gid = self.source.get_size()

    def execute(self):
        """TODO"""
        if not self.active:
            return

        for gid in self.source.get_spikes_immediate():
            if gid >= self.from_gid:
                if gid <= self.to_gid and gid % self.every == 0:
                    self.outfile.write("%.6f %d\n".encode()
                                       % (core.kernel.get_time(), gid))
