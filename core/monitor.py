# -*- coding: utf-8 -*-
"""TODO"""

import os

import core.core_global as core
from core.device import Device


class Monitor(Device):
    """TODO"""

    def __init__(self, filename):
        """TODO"""
        super().__init__()
        self.active = True
        self.filename = filename

        try:
            os.makedirs(os.path.dirname(self.filename), exist_ok=True)
            self.outfile = open(self.filename, 'wb')
        except OSError:
            core.logger.error("Can't open output file %s."
                              % self.filename)
            raise

    def flush(self):
        """TODO"""
        self.outfile.flush()
        os.fsync(self.outfile.fileno())

    def __del__(self):
        """TODO"""
        try:
            self.outfile.close()
        except AttributeError:
            pass
