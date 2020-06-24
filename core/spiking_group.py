# -*- coding: utf-8 -*-
"""TODO"""

import abc
from enum import Enum

import core.core_global as core
from core.core_definitions import (SimulationDelayTooSmallException, simulation_min_distributed_size,
                                   simulation_mindelay)
from core.logger import LogMessageType
from core.euler_trace import EulerTrace
from core.spike_delay import SpikeDelay


class NodeDistributionMode(Enum):
    AUTO = 1
    ROUNDROBIN = 2
    BLOCKLOCK = 3
    RANKLOCK = 4


class SpikingGroup(metaclass=abc.ABCMeta):
    """TODO"""
    unique_id_count = 0
    last_locked_rank = 0

    def __init__(self, n, mode=NodeDistributionMode.AUTO):
        """TODO"""
        self.size = n
        self.group_name = "SpikingGroup"
        self.state_vectors = {}

        self._pre_traces = []
        self._mid_traces = []
        self._post_traces = []

        self._mpi_size = core.mpicomm.size
        self._mpi_rank = core.mpicomm.rank

        self._locked_rank = 0
        self._locked_range = self._mpi_size
        self.rank_size = self._calculate_rank_size()
        self.delay = SpikeDelay(simulation_mindelay + 1)

        self._spikes = None
        self._clock_fun = None
        self.evolve_locally = True
        self.unique_id = SpikingGroup.unique_id_count
        SpikingGroup.unique_id_count += 1

        if self._mpi_size == 0:
            mode = NodeDistributionMode.ROUNDROBIN

        frac = self._calculate_rank_size(0) / simulation_min_distributed_size
        if mode == NodeDistributionMode.AUTO:
            if 0 <= frac < 1:
                mode = NodeDistributionMode.BLOCKLOCK
            else:
                mode = NodeDistributionMode.ROUNDROBIN

        if mode == NodeDistributionMode.BLOCKLOCK:
            self._lock_range(frac)
        elif mode == NodeDistributionMode.RANKLOCK:
            self._lock_range(0)
        elif mode == NodeDistributionMode.ROUNDROBIN:
            self._locked_rank = 0
            self._locked_range = self._mpi_size
            if self._mpi_size > 1:
                core.logger.notification("%s :: Size %d (ROUNDROBIN)"
                                         % (self.group_name,
                                            self.rank_size))
            else:
                core.logger.notification("%s :: Size %d"
                                         % (self.group_name,
                                            self.rank_size))

        self.evolve_locally = self.evolve_locally and (self.rank_size > 0)

    def _calculate_rank_size(self, rank=-1):
        """TODO"""
        if rank >= 0:
            comrank = rank
        else:
            comrank = self._mpi_rank

        if self._locked_rank <= comrank < self._locked_rank + self._locked_range:
            if comrank - self._locked_rank >= self.size % self._locked_range:
                return self.size // self._locked_range
            else:
                return self.size // self._locked_range + 1
        else:
            return 0

    def _lock_range(self, rank_fraction):
        """TODO"""
        self._locked_rank = SpikingGroup.last_locked_rank % self._mpi_size

        if rank_fraction == 0:
            self._locked_range = 1
            core.logger.notification(
                "%s :: Group will run on single rank only (RANKLOCK)"
                % self.group_name)
        else:
            free_ranks = self._mpi_size - SpikingGroup.last_locked_rank
            self._locked_range = int(rank_fraction * self._mpi_size + 0.5)
            if self._locked_range == 0:
                self._locked_range = 1
            if self._locked_range > free_ranks:
                self._locked_rank = 0
                free_ranks = self._mpi_size
                core.logger.msg(
                    "%s :: Not enough free ranks for RANGELOCK. Starting to "
                    "fill at zero again ..."
                    % self.group_name)

        rank = self._mpi_rank
        self.evolve_locally = ((rank >= self._locked_rank) and
                               (rank < (self._locked_rank +
                                        self._locked_range)))
        SpikingGroup.last_locked_rank = ((self._locked_rank +
                                          self._locked_range) % self._mpi_size)
        self.rank_size = self._calculate_rank_size()

        if self.evolve_locally:
            core.logger.notification(
                "%s :: Size %d (BLOCKLOCK [%d : %d]"
                % (self.group_name, self.rank_size, self._locked_rank,
                   self._locked_range + self._locked_rank - 1))
        else:
            core.logger.msg(
                "%s :: Passive on this rank (BLOCKLOCK [%d : %d]"
                % (self.group_name, self._locked_rank,
                   self._locked_range + self._locked_rank - 1),
                LogMessageType.VERBOSE)

    def set_clock(self, clock_fun):
        """TODO"""
        self._clock_fun = clock_fun
        self.delay.set_clock(clock_fun)

    def push_spike(self, spike):
        """TODO"""
        self._spikes.append(self.rank2global(spike))

    def get_spikes(self):
        """TODO"""
        return self.delay.get_spikes()

    def get_spikes_immediate(self):
        """TODO"""
        return self.delay.get_spikes_immediate()

    def get_state_vector(self, key, shape=None):
        """TODO"""
        if key in self.state_vectors:
            return self.state_vectors[key]
        else:
            # TODO
            pass

    def get_vector_size(self):
        """TODO"""
        return self.rank_size

    def get_size(self):
        """TODO"""
        return self.size

    def get_pre_size(self):
        """TODO"""
        return self.get_size()

    def get_post_size(self):
        """TODO"""
        return self.rank_size

    def get_pre_trace(self, tau):
        """TODO"""
        for tr in self._pre_traces:
            if tr.get_tau() == tau:
                return tr

        tr = EulerTrace(self.get_pre_size(), tau)
        self.add_pre_trace(tr)

        return tr

    def add_pre_trace(self, tr):
        """TODO"""
        if tr.val.size != self.get_pre_size():
            core.logger.warning("Trying to add as pre trace, but its size"
                                " does not match the SpikinGroup.",
                                LogMessageType.WARNING)
        self._pre_traces.append(tr)

    def get_post_trace(self, tau):
        """TODO"""
        for tr in self._post_traces:
            if tr.get_tau() == tau:
                return tr

        tr = EulerTrace(self.get_post_size(), tau)
        self.add_post_trace(tr)

        return tr

    def add_post_trace(self, tr):
        """TODO"""
        if tr.val.size != self.get_post_size():
            core.logger.warning("Trying to add as post trace, but its"
                                " size does not match the SpikinGroup.",
                                LogMessageType.WARNING)
        self._post_traces.append(tr)

    def set_delay(self, delay):
        """TODO"""
        if delay < simulation_mindelay:
            raise SimulationDelayTooSmallException()

        self.delay.set_delay(delay)

    def rank2global(self, i):
        """TODO"""
        return i * self._locked_range + (self._mpi_rank - self._locked_rank)

    def global2rank(self, i):
        """TODO"""
        return i // self._locked_range

    def evolve_traces(self):
        """TODO"""
        # Evolve presynaptic traces
        for tr in self._pre_traces:
            for spike in self.get_spikes():
                tr.inc(spike)
            tr.evolve()

        # Evolve postsynaptic traces
        for tr in self._post_traces:
            for spike in self.get_spikes_immediate():
                tr.inc(self.global2rank(spike))
            tr.evolve()

    def conditional_evolve(self):
        """TODO"""
        self._spikes = self.get_spikes_immediate()
        self._spikes.clear()
        if self.evolve_locally:
            self.evolve()

    @abc.abstractmethod
    def evolve(self):
        """TODO"""
        pass
