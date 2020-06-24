# -*- coding: utf-8 -*-
"""TODO"""
import os
import time

import numpy as np
from mpi4py import MPI

from core.logger import LogMessageType
from core.core_definitions import (simulation_mindelay, simulation_timestep)
from core.sync_buffer import SyncBuffer


def _progress_bar(fraction):
    """TODO"""
    bar = ""
    division = 4
    percent = 100 * fraction
    for i in range(100//division):
        if i < percent // division:
            bar += '='
        elif i == percent // division:
            bar += '>'
        else:
            bar += ' '

    print('[%s] %d%%\r' % (bar, percent), end='')


class System:
    """TODO"""

    def __init__(self, mpicomm, logger, directory, simulation_name, quiet):
        """TODO"""
        self.quiet = quiet
        self.out_dir = directory
        self.simulation_name = simulation_name
        self.progress_bar_update_interval = 1000

        self._mpicomm = mpicomm
        self._logger = logger

        self._mpi_size = mpicomm.size
        self._mpi_rank = mpicomm.rank

        self._sync_buffer = SyncBuffer(self._mpicomm)

        self._clock = 0
        self._devices = []
        self._checkers = []
        self._connections = []
        self._spiking_groups = []

        self.default_seed = 8010
        self.set_master_seed(self.default_seed)

        logger.notification("Starting Spaghetti Kernel")
        logger.msg("Simulation timestep is set to %.2es"
                   % simulation_timestep, LogMessageType.SETTINGS)

        if self._mpi_size > 0 and (self._mpi_size & (self._mpi_size - 1)):
            self._logger.msg("The number of processes is not a power of "
                             "two. This causes impaired performance or "
                             "even crashes in some MPI implementations.",
                             LogMessageType.WARNING, True)

    def _step(self):
        """TODO"""
        self._clock += 1

    def _run(self, start_time, stop_time, total_time, checking=False):
        """TODO"""
        if self.get_total_neurons() == 0:
            self._logger.warning("There are no units assigned to this rank")

        run_time = (stop_time - self._clock) * simulation_timestep

        self._logger.notification("Simulation triggered (run time = %.2fs) ..."
                                  % run_time)

        if self._clock == 0:
            self._logger.msg("On this rank: neurons %d, synapses %d"
                             % (self.get_total_neurons(),
                                self.get_total_synapses()),
                             LogMessageType.SETTINGS)
            if self._mpi_rank == 0:
                all_ranks_total_neurons = self._mpicomm.reduce(
                    self.get_total_neurons(), op=MPI.SUM)
                all_ranks_total_synapses = self._mpicomm.reduce(
                    self.get_total_synapses(), op=MPI.SUM)
                self._logger.msg("On all ranks: neurons %d, synapses %d"
                                 % (all_ranks_total_neurons,
                                    all_ranks_total_synapses),
                                 LogMessageType.SETTINGS)
            else:
                self._mpicomm.reduce(self.get_total_neurons(), op=MPI.SUM)
                self._mpicomm.reduce(self.get_total_synapses(), op=MPI.SUM)

        t_sim_start = time.process_time()

        while self._clock < stop_time:
            # Update progressbar
            if (self._mpi_rank == 0
                    and not self.quiet
                    and (self._clock % self.progress_bar_update_interval == 0
                         or self._clock == stop_time - 1)):
                fraction = ((self._clock - start_time + 1) *
                            simulation_timestep / total_time)
                _progress_bar(fraction)

            # Evolve neuron groups
            self._evolve()

            # Propagate spikes through connections and implement plasticity
            self._propagate()

            # Update internal state of connections
            self._evolve_connections()

            # Call monitors and recording devices
            self._execute_devices()

            # Run checker to break run if needed
            if checking:
                if not self._execute_checkers():
                    return False

            self._step()

            # Sync nodes
            if (self._mpi_size > 1
                    and self._clock % simulation_mindelay == 0):
                self._sync()

        elapsed = time.process_time() - t_sim_start
        self._logger.notification("Simulation finished. Elapsed wall time "
                                  "%.2fs" % elapsed)

        return True

    def _evolve(self):
        """TODO"""
        for spiking_group in self._spiking_groups:
            spiking_group.conditional_evolve()
        for device in self._devices:
            device.evolve()

    def _propagate(self):
        """TODO"""
        for connection in self._connections:
            connection.propagate()

    def _execute_devices(self):
        """TODO"""
        for device in self._devices:
            device.execute()

    def _execute_checkers(self):
        """TODO"""
        for i, checker in enumerate(self._checkers):
            if not checker.execute():
                self._logger.warning("Checker %d triggered abort of simulation" % i)
                return False

        return True

    def _evolve_connections(self):
        """TODO"""
        for spiking_group in self._spiking_groups:
            spiking_group.evolve_traces()
        for connection in self._connections:
            connection.evolve()

    def _sync(self):
        """TODO"""
        for spiking_group in self._spiking_groups:
            self._sync_buffer.push(spiking_group.delay,
                                   spiking_group.size)

        self._sync_buffer.null_terminate_send_buffer()
        self._sync_buffer.sync()

        for spiking_group in self._spiking_groups:
            self._sync_buffer.pop(spiking_group.delay,
                                  spiking_group.size)

    def run(self, simulation_time, checking=False):
        """TODO"""
        if simulation_time < 0.0:
            self._logger.error("Negative run time not allowed")
            return False

        start_time = self._clock
        stop_time = self._clock + int(simulation_time / simulation_timestep)

        return self._run(start_time, stop_time, simulation_time, checking)

    def run_chunk(self, chunk_time, interval_start, interval_end,
                  checking=False):
        """TODO"""
        start_time = int(interval_start / simulation_timestep)
        stop_time = self._clock + int(chunk_time / simulation_timestep)
        simulation_time = interval_end - interval_start

        return self._run(start_time, stop_time, simulation_time, checking)

    def get_clock(self):
        """TODO"""
        return self._clock

    def get_time(self):
        """TODO"""
        return self._clock * simulation_timestep

    def get_total_neurons(self):
        """TODO"""
        total = 0
        for spiking_group in self._spiking_groups:
            total += spiking_group.rank_size

        return total

    def get_total_synapses(self):
        """TODO"""
        total = 0
        for connection in self._connections:
            total += connection.get_nonzero()

        return total

    def register_spiking_group(self, spiking_group):
        """TODO"""
        self._spiking_groups.append(spiking_group)
        spiking_group.set_clock(self.get_clock())

    def register_connection(self, connection):
        """TODO"""
        self._connections.append(connection)

    def register_device(self, device):
        """TODO"""
        self._devices.append(device)

    def register_checker(self, checker):
        """TODO"""
        self._checkers.append(checker)

    def fn(self, name, extension, index=None):
        """TODO"""
        if index is not None:
            name = (name + str(index) + os.path.extsep + str(self._mpi_rank) +
                    os.path.extsep + extension)
        else:
            name = (name + os.path.extsep + str(self._mpi_rank) +
                    os.path.extsep + extension)

        return os.path.join(self.out_dir, name)

    def set_master_seed(self, seed=None):
        """TODO"""
        if seed is None:
            seed = int(time.time())

        seed_multiplier = 257
        self.rank_master_seed = seed_multiplier * seed * (self._mpi_rank + 1)
        self._logger.msg("Seeding this rank with master seed %d" %
                         self.rank_master_seed, LogMessageType.NOTIFICATION)
        self.rng = np.random.RandomState(seed=self.rank_master_seed)

    def get_seed(self):
        """TODO"""
        return self.rank_master_seed

    def get_rng(self):
        """TODO"""
        return self.rng
