#!/usr/bin/env python
"""This script is meant to analyze the effect of the temperature T on the change in the synaptic weights.

We start with 20 synapses/per branch for each of the 8 assemblies (8 branches have connections to 20 different
input neurons; the remaining 4 branches are not connected). We set the weights to the maximum and simulate for
t seconds, where 7 assemblies are active during simulation and 1 assembly only spikes with a background rate
of 1Hz. We are interessted in the weight change of that one branch that has connections to the assembly that
is silent during simulation.
"""

import os
import time

import numpy as np

from core import core_global as core
from core.spike_monitor import SpikeMonitor
from core.voltage_monitor import VoltageMonitor
from core.weight_matrix_monitor import WeightMatrixMonitor
from layers.rewiring_connection import RewiringConnection
from models.mc_lif_group import McLifGroup
from models.poisson_pattern_group import PoissonPatternGroup
from utils import utils as utils


def main(args):
    trial = args[0]
    config = args[1]
    input_params = config["input_parameters"]
    connection_params = config["connection_parameters"]
    neuron_params = config["neuron_parameters"]

    # Directory for simulation results and log files.
    output_directory = os.path.join("results", "rewiring_ex9", "T" + str(connection_params["T"]),
                                    time.strftime("%y%m%d_%H%M%S"), str(trial), "data")

    # Initialize the simulation environment.
    core.init(directory=output_directory)

    # Write config file to the output directory.
    utils.write_configuration(os.path.join(output_directory, "..", "config_rewiring_ex9.yaml"), config)

    # Set the random seed.
    core.kernel.set_master_seed(config["master_seed"])

    # Create input neurons.
    inp = PoissonPatternGroup(input_params["num_inputs"], input_params["rate"], input_params["rate_bg"],
                              params=input_params)

    # Create the neuron.
    neuron = McLifGroup(1, neuron_params["num_branches"], neuron_params)

    # Connect input to neuron.
    conn = RewiringConnection(inp, neuron, neuron.branch.syn_current, connection_params)

    # Set the weights.
    size = (neuron_params["num_branches"], input_params["num_inputs"])
    theta_ini = connection_params["theta_ini"]
    theta = np.full(size, theta_ini, np.float32)

    assembly_neurons_idc = [i * input_params["assembly_size"] + np.arange(input_params["assembly_size"])
                            for i in range(input_params["num_assemblies"]+1)]

    for idc, row in zip(assembly_neurons_idc, theta):
        row[core.kernel.rng.choice(idc, connection_params["n_syn_start"], replace=False)] = \
            core.kernel.rng.uniform(low=connection_params["w_ini_min"],
                                    high=connection_params["w_ini_max"],
                                    size=connection_params["n_syn_start"])

    conn.set_weights(theta)

    # Create some monitors which will record the simulation data.
    WeightMatrixMonitor(conn, core.kernel.fn("weights", "dat"),
                        interval=config["sampling_interval_weights"])
    SpikeMonitor(neuron, core.kernel.fn("output", "ras"))
    sm_inp = SpikeMonitor(inp, core.kernel.fn("input", "ras"))
    vm_nrn = VoltageMonitor(neuron.soma, 0, core.kernel.fn("soma", "mem"))
    vm_br = []
    for i in range(neuron_params["num_branches"]):
        vm_br.append(VoltageMonitor(neuron.branch, (i, 0), core.kernel.fn("branch", "mem", i)))

    # Now simulate the model.
    simulation_time = config["simulation_time"]
    core.kernel.run_chunk(20.0, 0, simulation_time)

    sm_inp.active = False
    vm_nrn.active = False
    for vm in vm_br:
        vm.active = False
    core.kernel.run_chunk(simulation_time - 40, 0, simulation_time)

    sm_inp.active = True
    vm_nrn.active = True
    for vm in vm_br:
        vm.active = True
    core.kernel.run_chunk(20.0, 0, simulation_time)


if __name__ == '__main__':
    import copy
    from scoop import futures

    # Load the configuration file.
    config = utils.load_configuration("config_rewiring_ex9.yaml")

    configs = []
    num_trials = 25
    for trial in range(num_trials):
        config["master_seed"] = 10 * (trial + 1)
        configs.append(copy.deepcopy(config))

    r = list(futures.map(main, [[trial, config] for trial, config in enumerate(configs)]))
