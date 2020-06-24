# -*- coding: utf-8 -*-
"""TODO"""

import numpy as np

import core.core_global as core
from core.core_definitions import simulation_timestep
from core.spiking_group import SpikingGroup


class PoissonPatternGroup(SpikingGroup):
    """TODO"""

    def __init__(self, size, rate=1.0, rate_bg=1.0, params={}):
        """TODO"""
        super().__init__(size)

        core.kernel.register_spiking_group(self)

        if self.evolve_locally:
            self.set_rate(rate)
            self.set_bg_rate(rate_bg)
            self.pattern_delay = params.get("pattern_delay", 0.1)
            self.pattern_duration = params.get("pattern_duration", 0.4)
            self.num_assemblies = params.get("num_assemblies", 2)
            self.assembly_size = params.get("assembly_size", 4)
            self.neuron_activation_probability = params.get(
                "neuron_activation_probability", 1.0)
            self.num_simultaneous_assemblies = params.get(
                "num_simultaneous_assemblies", 1)

            self.timesteps_delay = int(self.pattern_delay / simulation_timestep)
            self.timesteps_pattern = int(self.pattern_duration /
                                         simulation_timestep)
            self.pattern_start = self.timesteps_delay
            self.pattern_end = self.timesteps_delay + self.timesteps_pattern

            self.set_assembly_neurons_idc(
                [i * self.assembly_size + np.arange(self.assembly_size)
                 for i in range(self.num_assemblies)])
            self.set_assemblies(np.arange(self.num_assemblies))

            self.rng = np.random.RandomState(seed=core.kernel.get_seed())

    def evolve(self):
        """TODO"""
        clock = core.kernel.get_clock()
        if clock == self.pattern_end or clock == 0:
            self.idc_inactive_neurons = []
            self.choose_active_assemblies()
            self.choose_inactive_neurons_idc()
            self.set_non_assembly_neurons_idc()
            self.pattern_start = clock + self.timesteps_delay
            self.pattern_end = clock + self.timesteps_delay + \
                self.timesteps_pattern

        if clock >= self.pattern_start and clock < self.pattern_end:
            for assembly, idx_inactive_neurons in zip(
                    self.active_assemblies, self.idc_inactive_neurons):

                mask = self.rng.uniform(
                    size=self.assembly_size) < self.lambd * simulation_timestep
                spikes = self.assembly_neurons_idc[assembly][mask]

                inactive_neurons = idx_inactive_neurons + \
                    int(assembly * self.assembly_size)
                mask = np.isin(spikes, inactive_neurons, invert=True)

                for spike in spikes[mask]:
                    self.push_spike(spike)

                # The number of active neurons per assembly presentation should
                # be constant. Therefore we choose 'inactive_neurons_size'
                # neurons at random from all non assembly neurons.
                mask = self.rng.uniform(
                    size=self.inactive_neurons_size) < \
                    self.lambd * simulation_timestep
                spikes = self.non_assembly_neurons_idc[mask]
                for spike in spikes:
                    self.push_spike(spike)

        spikes_bg = np.flatnonzero(self.rng.uniform(
            size=self.rank_size) < self.lambd_bg * simulation_timestep)
        for spike in spikes_bg:
            self.push_spike(spike)

    def set_rate(self, rate):
        """TODO"""
        self.lambd = 1.0/(1.0/rate - simulation_timestep)

    def set_bg_rate(self, rate):
        """TODO"""
        self.lambd_bg = 1.0/(1.0/rate - simulation_timestep)

    def set_assemblies(self, assemblies):
        """TODO"""
        if isinstance(assemblies, (list, tuple, np.ndarray)):
            self.assemblies = assemblies
        else:
            self.assemblies = [assemblies]

    def set_assembly_neurons_idc(self, idc):
        """TODO"""
        self.assembly_neurons_idc = idc

    def set_non_assembly_neurons_idc(self):
        """TODO"""
        self.non_assembly_neurons_idc = self.rng.choice(np.setdiff1d(
            np.arange(self.rank_size), [self.assembly_neurons_idc[assembly] for
                                        assembly in self.active_assemblies]),
            self.inactive_neurons_size, replace=False)

    def choose_active_assemblies(self):
        """TODO"""
        self.active_assemblies = self.rng.choice(
            self.assemblies, self.num_simultaneous_assemblies, replace=False)

    def choose_inactive_neurons_idc(self):
        """TODO"""
        size = round(self.assembly_size *
                     (1 - self.neuron_activation_probability))
        self.inactive_neurons_size = size * self.num_simultaneous_assemblies
        for i in range(self.num_simultaneous_assemblies):
            self.idc_inactive_neurons.append(self.rng.choice(
                self.assembly_size, size, replace=False))

    def get_rate(self):
        """TODO"""
        return self.lambd

    def get_bg_rate(self):
        """TODO"""
        return self.lambd_bg
