# -*- coding: utf-8 -*-
"""TODO"""

simulation_timestep = 1e-3

simulation_mindelay = 8

simulation_min_distributed_size = 16

# Error codes:
# -1: Cannot write/open logger logfile.


class SimulationDelayTooSmallException(Exception):
    """TODO"""

    def __init__(self, message="Axonal delay can not be shorter than "
                               "'simulation_mindelay'."):
        """TODO"""
        super().__init__(message)
