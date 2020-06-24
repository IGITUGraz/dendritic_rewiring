# -*- coding: utf-8 -*-
"""TODO"""

import numpy as np

import core.core_global as core
from core.core_definitions import simulation_timestep
from core.monitor import Monitor


class WeightMatrixMonitor(Monitor):
    """TODO"""

    def __init__(self, source, filename, interval=1.0):
        """TODO"""
        super().__init__(filename)

        core.kernel.register_device(self)

        self.source = source
        self.stepsize = int(interval / simulation_timestep)

        if self.stepsize < 1:
            self.stepsize = 1

        self.outfile.write("# Recording with a sampling interval of %.2fs at a"
                           " timestep of %.2es\n".encode()
                           % (interval, simulation_timestep))
        self.outfile.write("# The shape (post size, pre size) of the matrix is"
                           " {0}\n".format(self.source.w.shape).encode())

    def execute(self):
        """TODO"""
        if not self.active:
            return

        if self.source.get_destination().evolve_locally:
            if core.kernel.get_clock() % self.stepsize == 0:
                np.savetxt(self.outfile, self.source.w, fmt="%.6f",
                           header="%.6f" % (core.kernel.get_time()))
