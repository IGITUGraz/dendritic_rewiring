#!/usr/bin/env python
"""Sensitivity analysis for rewiring_ex2."""

import os
import time
from functools import partial

import utils as utils
from spaghetti import spaghetti_global as spaghetti
from spaghetti.mc_lif_group import McLifGroup
from spaghetti.poisson_pattern_group import PoissonPatternGroup
from spaghetti.rewiring_connection import RewiringConnection
from spaghetti.spike_monitor import SpikeMonitor
from spaghetti.voltage_monitor import VoltageMonitor
from spaghetti.weight_matrix_monitor import WeightMatrixMonitor


def main(args, param, change):
    trial = args[0]
    config = args[1]
    input_params = config["input_parameters"]
    connection_params = config["connection_parameters"]
    neuron_params = config["neuron_parameters"]

    # Directory for simulation results and log files.
    output_directory = os.path.join(
        os.path.sep, "calc", os.getenv("USER"), "data", "dendritic_rewiring",
        "rewiring_ex2", "sensitivity_analysis", param, change, time.strftime("%y%m%d_%H%M%S"), str(trial),
        "data")

    # Initialize the simulation environment.
    spaghetti.spaghetti_init(directory=output_directory)

    # Write config file to the output directory.
    utils.write_configuration(os.path.join(output_directory, "..", "config_rewiring_ex2.yaml"), config)

    # Set the random seed.
    spaghetti.kernel.set_master_seed(config["master_seed"])

    # Create input neurons.
    inp = PoissonPatternGroup(input_params["num_inputs"], input_params["rate"], input_params["rate_bg"],
                              params=input_params)

    # Create the neuron.
    neuron = McLifGroup(1, neuron_params["num_branches"], neuron_params)

    # Connect input to neuron.
    conn = RewiringConnection(inp, neuron, neuron.branch.syn_current, connection_params)

    # Create some monitors which will record the simulation data.
    WeightMatrixMonitor(conn, spaghetti.kernel.fn("weights", "dat"),
                        interval=config["sampling_interval_weights"])
    SpikeMonitor(neuron, spaghetti.kernel.fn("output", "ras"))
    sm_inp = SpikeMonitor(inp, spaghetti.kernel.fn("input", "ras"))
    vm_nrn = VoltageMonitor(neuron.soma, 0, spaghetti.kernel.fn("soma", "mem"))
    vm_br = []
    for i in range(neuron_params["num_branches"]):
        vm_br.append(VoltageMonitor(neuron.branch, (i, 0), spaghetti.kernel.fn("branch", "mem", i)))

    # Now simulate the model.
    simulation_time = config["simulation_time"]
    spaghetti.kernel.run_chunk(20.0, 0, simulation_time)

    sm_inp.active = False
    vm_nrn.active = False
    for vm in vm_br:
        vm.active = False
    spaghetti.kernel.run_chunk(simulation_time - 40, 0, simulation_time)

    sm_inp.active = True
    vm_nrn.active = True
    for vm in vm_br:
        vm.active = True
    spaghetti.kernel.run_chunk(20.0, 0, simulation_time)


if __name__ == '__main__':
    import copy
    from scoop import futures

    # Load the configuration file.
    config = utils.load_configuration("experiments/config_rewiring_ex2.yaml")

    # Change a specific parameter.
    param = "plateau_duration_max"
    delta_param = +0.1  # +-10 %

    # params = config["connection_parameters"]
    params = config["neuron_parameters"]["branch_parameters"]

    value = params[param]
    if value >= 0:
        params[param] += delta_param * value
    else:
        params[param] += -delta_param * value

    if delta_param >= 0:
        change = "+" + str(abs(delta_param) * 100)
    else:
        change = "-" + str(abs(delta_param) * 100)

    configs = []
    num_trials = 25
    for trial in range(num_trials):
        config["master_seed"] = 10 * (trial + 1)
        configs.append(copy.deepcopy(config))

    partial_main = partial(main, param=param, change=change)
    r = list(futures.map(partial_main, [[trial, config] for trial, config in enumerate(configs)]))
