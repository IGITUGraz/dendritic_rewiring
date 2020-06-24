# -*- coding: utf-8 -*-
"""TODO"""

import os
import sys

from mpi4py import MPI

from core.logger import Logger
from core.system import System

logger = None
kernel = None
mpicomm = None


def init(directory='.', simulation_name='default', logfile_prefix='default', quiet=False):
    """TODO"""
    # Initialize MPI and logger.
    _mpi_init(directory, logfile_prefix)

    # Initialize kernel.
    _kernel_init(directory, simulation_name, quiet)


def abort(errorcode):
    """TODO"""
    del logger
    mpicomm.Abort(errorcode)
    sys.exit(errorcode)


def free():
    """TODO"""
    _kernel_free()
    _mpi_free()


def _kernel_free():
    """TODO"""
    global kernel

    del kernel


def _mpi_free():
    """TODO"""
    global mpicomm, logger

    del logger
    del mpicomm


def _mpi_init(directory, logfile_prefix):
    """TODO"""
    global mpicomm, logger

    mpicomm = MPI.COMM_WORLD
    local_rank = mpicomm.rank

    try:
        suffix = '.%d.log' % local_rank
        logfile_name = os.path.join(directory, logfile_prefix + suffix)
        logger = Logger(logfile_name, local_rank)
    except IOError:
        print("Cannot proceed without log file. Exiting all ranks ...")
        abort(-1)


def _kernel_init(directory, simulation_name, quiet):
    """TODO"""
    global kernel

    kernel = System(mpicomm, logger, directory, simulation_name, quiet)
