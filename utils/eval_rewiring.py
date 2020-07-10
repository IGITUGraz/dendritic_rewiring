# /usr/bin/env python

import os

import utils as utils
from spaghetti import spaghetti_global as spaghetti
from spaghetti.mc_lif_group import McLifGroup
from spaghetti.poisson_pattern_group import PoissonPatternGroup
from spaghetti.rewiring_connection import RewiringConnection
from spaghetti.spike_monitor import SpikeMonitor
from spaghetti.voltage_monitor import VoltageMonitor


def main(simulation_time, experiment, sim_date, assemblies,
         assembly_neurons_idc, config, output_directory):

    input_params = config["input_parameters"]
    connection_params = config["connection_parameters"]
    neuron_params = config["neuron_parameters"]
    simulation_time_per_pattern = (input_params["pattern_duration"] +
                                   input_params["pattern_delay"])

    # Initialize the simulation environment.
    spaghetti.spaghetti_init(directory=output_directory)

    # Set the random seed.
    spaghetti.kernel.set_master_seed(config["master_seed"])

    # Create input neurons.
    inp = PoissonPatternGroup(input_params["num_inputs"], input_params["rate"],
                              input_params["rate_bg"], params=input_params)

    # Create the neuron.
    neuron = McLifGroup(1, neuron_params["num_branches"], neuron_params)

    # Connect input to neuron.
    conn = RewiringConnection(inp, neuron, neuron.branch.syn_current,
                              connection_params)
    conn.learn = False

    # Create some monitors which will record the simulation data.
    SpikeMonitor(neuron, spaghetti.kernel.fn("test_output", "ras"))
    SpikeMonitor(inp, spaghetti.kernel.fn("test_input", "ras"))
    VoltageMonitor(neuron.soma, 0, spaghetti.kernel.fn("test_soma", "mem"),
                   paste_spikes=True, paste_spikes_from=neuron)
    for i in range(neuron_params["num_branches"]):
        VoltageMonitor(neuron.branch, (i, 0),
                       spaghetti.kernel.fn("test_branch", "mem", i))

    # Now simulate the model.
    conn.set_weights(weights_pre)
    for assembly in assemblies:
        inp.set_assemblies(assembly)
        inp.set_assembly_neurons_idc(assembly_neurons_idc)
        spaghetti.kernel.run_chunk(
            simulation_time_per_pattern, 0, simulation_time)

    conn.set_weights(weights_post)
    for assembly in assemblies:
        inp.set_assemblies(assembly)
        inp.set_assembly_neurons_idc(assembly_neurons_idc)
        spaghetti.kernel.run_chunk(
            simulation_time_per_pattern, 0, simulation_time)

    spaghetti.kernel.run_chunk(input_params["pattern_delay"], 0,
                               simulation_time)


if __name__ == '__main__':
    import numpy as np

    experiment = "rewiring_ex1"
    sim_date = "181228_163308-19"
    master_seed = 13
    simulation_time = 3.2
    assemblies = [0, 1, 2]
    assembly_neurons_idc = [i * 40 + np.arange(40) for i in assemblies[0:2]]
    assembly_neurons_idc += [4 * 40 + np.arange(40)]

    input_directory = os.path.join("results", experiment, sim_date, "data")

    # Load the configuration file.
    config = utils.load_configuration(os.path.join(
        input_directory, "..", "config_" + experiment + ".yaml"))
    num_branches = config["neuron_parameters"]["num_branches"]
    config["master_seed"] = master_seed
    if experiment == "rewiring_ex3":
        assembly_neurons_idc = []
        assembly_neurons_idc = np.loadtxt(os.path.join(input_directory,
                                          "assembly_neurons_idc"),
                                          dtype=np.int)
    else:
        np.save(os.path.join(input_directory, "assembly_neurons_idc"),
                assembly_neurons_idc)

    # Load the weights.
    header_lenght = 3
    with open(os.path.join(input_directory, "weights.0.dat"), "rb") as f:
        lines = f.readlines()
    weights_pre = np.loadtxt(lines[header_lenght:num_branches + header_lenght])
    weights_post = np.loadtxt(lines[-num_branches:])

    main(simulation_time, experiment, sim_date, assemblies,
         assembly_neurons_idc, config, input_directory)
