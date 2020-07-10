# /usr/bin/env python

import glob
import os
import subprocess

import plotting.configure_seaborn as cs
import matplotlib.colors as mcolors
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import utils as utils
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.axes_grid1 import ImageGrid, make_axes_locatable
from scipy.special import expit

sns.set(context='paper', style='ticks', rc=cs.rc_params)


def _load_configuration(input_directory, experiment, keys):
    config = utils.load_configuration(os.path.join(
        input_directory, "..", "config_" + experiment + ".yaml"))

    if len(keys) > 1:
        return config[keys[0]][keys[1]]
    else:
        return config[keys[0]]


def _load_simulation_results(input_directory, experiment, key, gid=None,
                             header_lenght=3):
    if key == "input_spike_times":
        data = np.loadtxt(os.path.join(input_directory, "input.0.ras"))
    if key == "input_spike_times_test":
        data = np.loadtxt(os.path.join(input_directory, "test_input.0.ras"))
    elif key == "output_spike_times":
        data = np.loadtxt(os.path.join(input_directory, "output.0.ras"))
    elif key == "output_spike_times_test":
        data = np.loadtxt(os.path.join(input_directory, "test_output.0.ras"))
    elif key == "weights_end":
        num_branches = _load_configuration(
            input_directory, experiment, ["neuron_parameters", "num_branches"])
        with open(os.path.join(input_directory, "weights.0.dat"), "rb") as f:
            lines = f.readlines()
        data = np.loadtxt(lines[-num_branches:])
    elif key == "weights_start":
        num_branches = _load_configuration(
            input_directory, experiment, ["neuron_parameters", "num_branches"])
        with open(os.path.join(input_directory, "weights.0.dat"), "rb") as f:
            lines = f.readlines()
        data = np.loadtxt(lines[header_lenght:num_branches + header_lenght])
    elif key == "weights":
        with open(os.path.join(input_directory, "weights.0.dat"), "rb") as f:
            lines = f.readlines()
        data = np.loadtxt(lines)
    elif key == "branch_mem":
        data = np.loadtxt(os.path.join(input_directory, "branch" + str(gid) +
                                       ".0.mem"))
    elif key == "branch_mem_test":
        data = np.loadtxt(os.path.join(input_directory, "test_branch" +
                                       str(gid) + ".0.mem"))
    elif key == "soma_mem":
        data = np.loadtxt(os.path.join(input_directory, "soma.0.mem"))
    elif key == "soma_mem_test":
        data = np.loadtxt(os.path.join(input_directory, "test_soma.0.mem"))
    elif key == "plateau_duration":
        data = np.loadtxt(os.path.join(input_directory, "branch0.0.pla"))

    return data


def _get_represented_assemblies(weights, assembly_idc, min_summed_weight,
                                min_num_synapses):
    # w = utils.reject_outliers(weights)
    w = weights

    num_synapses = []
    summed_weight = []
    for i, assembly_idx in enumerate(assembly_idc):
        num_synapses.append(sum(np.heaviside(w[assembly_idx], 0)))
        summed_weight.append(
            sum(w[assembly_idx][w[assembly_idx] > 0], 0))

    idc = np.where(
        (np.asanyarray(summed_weight) >= min_summed_weight) &
        (np.asanyarray(num_synapses) >= min_num_synapses))[0].tolist()

    return idc, num_synapses, summed_weight


def _get_start_and_stop_times_of_test_patterns(experiment, input_directory):
    num_assemblies = _load_configuration(input_directory, experiment, ["input_parameters", "num_assemblies"])
    num_test_patterns_per_assembly = _load_configuration(input_directory, experiment, ["input_parameters",
                                                         "num_test_patterns_per_assembly"])
    num_patterns_per_assembly = _load_configuration(input_directory, experiment, ["input_parameters",
                                                    "num_patterns_per_assembly"])
    pattern_delay = _load_configuration(input_directory, experiment, ["input_parameters", "pattern_delay"])
    pattern_duration = _load_configuration(input_directory, experiment, ["input_parameters",
                                           "pattern_duration"])

    # Start and end times of test patterns.
    duration_test_patterns = (pattern_delay + pattern_duration) * \
        num_test_patterns_per_assembly * num_assemblies
    times_start = np.array(
        [((pattern_delay + pattern_duration) *
         num_patterns_per_assembly +
         i * (pattern_delay + pattern_duration) *
         (num_patterns_per_assembly +
         (num_assemblies * num_test_patterns_per_assembly)))
         for i in range(num_assemblies)])
    times_end = times_start + duration_test_patterns

    return times_start, times_end


def _compute_statistics(experiment, input_directory, logfile, min_summed_weight=50, min_num_synapses=10):
    num_branches = _load_configuration(input_directory, experiment, ["neuron_parameters", "num_branches"])
    num_assemblies = _load_configuration(input_directory, experiment, ["input_parameters", "num_assemblies"])
    assembly_size = _load_configuration(input_directory, experiment, ["input_parameters", "assembly_size"])

    weights_end = _load_simulation_results(input_directory, experiment, "weights_end")

    if experiment == "rewiring_ex5" or experiment == "rewiring_ex6":
        logfile.write("  Overlap:\n".encode())

        assembly_neurons_idc = np.loadtxt(os.path.join(input_directory, "assembly_neurons_idc"),
                                          dtype=np.int)
        overlap = np.zeros((len(assembly_neurons_idc), len(assembly_neurons_idc)))
        for i in range(len(assembly_neurons_idc)):
            for j in range(len(assembly_neurons_idc)):
                overlap[i, j] = np.intersect1d(
                    assembly_neurons_idc[i],
                    assembly_neurons_idc[j]).size / assembly_size

        np.savetxt(logfile, overlap, fmt="%9.3f")
        idc = np.triu_indices_from(overlap, k=1)
        logfile.write("\n  Mean overlap: {0:0.3f} var: {1:0.3f}\n\n".format(
            np.mean(overlap[idc]), np.var(overlap[idc])).encode())

    else:
        assembly_neurons_idc = np.split(np.arange(num_assemblies * assembly_size), num_assemblies)

    assemblies = []
    num_synapses_assembly = []
    summed_weight_assembly = []
    num_synapses = []
    summed_weight = []
    num_assemblies_per_branch = np.zeros(num_branches)
    for i in range(num_branches):
        logfile.write("  Branch: {0}\n".format(i).encode())

        idc, num_synapses, summed_weight = _get_represented_assemblies(
            weights_end[i], assembly_neurons_idc, min_summed_weight,
            min_num_synapses)

        if idc:
            assemblies.append(idc)
            num_assemblies_per_branch[i] = len(idc)
            logfile.write("    Assembly: {0}, N_syn: {1}, sum(w): {2}\n"
                          .format(assemblies[-1], num_synapses,
                                  [round(x) for x in summed_weight])
                          .encode())
            num_synapses_assembly.append([num_synapses[a]
                                          for a in assemblies[-1]])
            summed_weight_assembly.append([summed_weight[a]
                                          for a in assemblies[-1]])
        else:
            logfile.write("    Assembly: {0}, N_syn: {1}, sum(w): {2}\n"
                          .format(" ", num_synapses,
                                  [round(x) for x in summed_weight])
                          .encode())

    num_represented_assemblies = len(np.unique([a for l in assemblies
                                                for a in l]))
    logfile.write("\n  Represented assemblies {0}/{1}, {2}\n".format(
        num_represented_assemblies, num_assemblies, assemblies).encode())

    logfile.write("  Number of synapses per assembly" " {0:0.3f} SD:"
                  " {1:0.3f}\n".format(
                      np.mean([n for l in num_synapses_assembly for n in l]),
                      np.std([n for l in num_synapses_assembly for n in l]))
                  .encode())

    logfile.write("  Summed weight per assembly"
                  " {0:0.3f} SD: {1:0.3f}\n\n".format(
                      np.mean([s for l in summed_weight_assembly for s in l]),
                      np.std([s for l in summed_weight_assembly for s in l]))
                  .encode())

    if experiment == "rewiring_ex5" or experiment == "rewiring_ex6":
        idc = np.triu_indices_from(overlap, k=1)
        return (num_represented_assemblies, overlap, num_assemblies_per_branch)
    else:
        return num_represented_assemblies, num_assemblies_per_branch


def _plot_num_represented_assemblies_over_act_probability(a, m, s, plots_directory):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    e = ax1.errorbar(range(len(m)), m, s, fmt='o', ms=3, clip_on=False)

    for bar in e[1]:
        bar.set_clip_on(False)
    for bar in e[2]:
        bar.set_clip_on(False)

    ax1.set_xlim([0, len(a) - 1])
    ax1.set_ylim([0, 8])
    ax1.set_xticklabels("%.1f" % f for f in np.linspace(np.max(a), np.min(a), len(a)))
    ax1.set_yticklabels("%d" % f for f in range(0, 10, 2))
    ax1.set_ylabel(r"# of represented assemblies")
    ax1.set_xlabel("Neuron activation probability")

    ax1.spines["bottom"].set_position(("data", -1.5))
    ax1.spines["left"].set_position(("data", -0.1))

    plt.tight_layout()
    fname = os.path.join(plots_directory, "..", "..", "num-represented-assemblies")
    fig.savefig(fname + ".pdf", pad_inches=0.01)
    subprocess.call(["pdftops", "-eps", fname + ".pdf", fname + ".eps"])
    plt.close(fig)


def _plot_weight_evolution_selected_branches(experiment, input_directory, plots_directory, branches=[0, 1, 2],
                                             patterns=[0, 1, 2], subsample=20, w_thr=7.0, linewidth=0.5,
                                             use_colors=True):
    colors = sns.color_palette().as_hex()
    if use_colors:
        colors_p = ['#9db2d5', '#e7a783', '#a2d0ad', '#c4c5c4']
    else:
        colors_p = ['#c4c5c4']
        colors = ['k']

    num_branches = _load_configuration(input_directory, experiment,
                                       ["neuron_parameters", "num_branches"])
    simulation_time = _load_configuration(input_directory, experiment,
                                          ["simulation_time"])
    w_max = _load_configuration(input_directory, experiment,
                                ["connection_parameters", "w_max"])
    sampling_interval_weights = _load_configuration(
        input_directory, experiment, ["sampling_interval_weights"])

    weights = _load_simulation_results(input_directory, experiment, "weights")

    if (experiment == "rewiring_ex3" or experiment == "rewiring_ex6"):
        t_start, t_end = _get_start_and_stop_times_of_test_patterns(experiment, input_directory)

        exclude_indc = []
        for ts, te in zip(t_start, t_end):
            exclude_indc += list(range(int(ts / sampling_interval_weights),
                                       int(te / sampling_interval_weights)))
        effective_simulation_time = simulation_time - np.sum(np.array(t_end) - np.array(t_start))

    else:
        exclude_indc = []
        effective_simulation_time = simulation_time

    cs.set_figure_size(176 + 7, 71 + 7)
    fig = plt.figure()
    gs = GridSpec(len(branches), 1)

    for i, (b, p) in enumerate(zip(branches, patterns)):
        ax = plt.subplot(gs[i])
        w = np.delete(weights[range(b, weights.shape[0], num_branches)],
                      exclude_indc, axis=0)
        w[w < 0] = 0
        w = w[::subsample]

        ax.plot(w[:, w[-1, :] < w_thr], color=colors_p[-1], alpha=1.0,
                linewidth=linewidth)

        if w[:, w[-1, :] >= w_thr].size:
            ax.plot(w[:, w[-1, :] >= w_thr], color=colors_p[p % len(colors_p)],
                    alpha=1.0, linewidth=linewidth)
            ax.plot(np.mean(w[:, w[-1, :] >= w_thr], axis=1),
                    color=colors[p % len(colors)], linewidth=1.0)
        ax.set_xlim([0, effective_simulation_time / (subsample *
                     sampling_interval_weights)])
        ax.set_xticks(np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], 5))
        ax.set_xticklabels("%.1f"
                           % (f * subsample * sampling_interval_weights)
                           for f in np.linspace(ax.get_xlim()[0],
                                                ax.get_xlim()[1], 5))
        ax.set_ylim([0, w_max])
        ax.set_yticks(np.arange(0, w_max + 1, 4))
        ax.set_yticklabels("%d" % f for f in ax.get_yticks())
        ax.set_ylabel(r"$w_{%di}$ [nA]" % (b + 1))

        if b == branches[-1]:
            ax.set_xlabel(r"$t$ [s]")
            ax.set_xticklabels("%.1f"
                               % (f * subsample * sampling_interval_weights)
                               for f in ax.get_xticks())
        else:
            ax.set_xticks([])

    plt.tight_layout()
    name = "weights"
    for b in branches:
        name += "-b%d" % b
    fname = os.path.join(plots_directory, name)
    fig.savefig(fname + ".pdf", pad_inches=0.01)
    subprocess.call(["pdftops", "-eps", fname + ".pdf", fname + ".eps"])
    plt.close(fig)


def main(experiment, sim_date):

    make_plots = False
    # make_plots = True
    comp_stats = not make_plots

    # Set this if `make_plots` is set to `True`.
    branches = [0, 1, 2]  # What branches to plot
    patterns = [0, 1, 2]  # This is just to get the color assigned to the assembly

    # Directories of simulation results and log files.
    input_directories = sorted(glob.iglob(os.path.join("results", experiment, sim_date, "data")))

    paths = {}
    for input_directory in input_directories:
        date = input_directory.split("/")[-3]
        logfile_path = os.path.join("results", experiment, date, "stats.txt")

        if logfile_path in paths:
            paths[logfile_path].append(input_directory)
        else:
            paths[logfile_path] = [input_directory]

    experiment = experiment.split("/")[0]

    neuron_activation_probability = []
    mean_num_represented_assemblies = []
    std_num_represented_assemblies = []
    for logfile_path, input_directories in paths.items():
        if comp_stats:
            logfile = open(logfile_path, "wb")

        if experiment == "rewiring_ex4":
            neuron_activation_probability.append(_load_configuration(
                          input_directories[0], experiment,
                          ["input_parameters",
                           "neuron_activation_probability"]))

        overlaps = []
        num_assemblies_per_branch = []
        num_represented_assemblies = []
        for trial, input_directory in enumerate(input_directories):

            if os.stat(os.path.join(input_directory,
                                    "weights.0.dat")).st_size == 0:
                continue

            plots_directory = os.path.join(input_directory, "..", "plots")
            if not os.path.exists(plots_directory):
                os.makedirs(plots_directory)

            if comp_stats and experiment != "test_plateau_duration":
                # Compute some statistics.
                logfile.write("Trial {0} (Data path: {1}):\n".format(
                    trial, input_directory).encode())

                result = _compute_statistics(experiment, input_directory,
                                             logfile)
                if experiment == "rewiring_ex5" or experiment == "rewiring_ex6":
                    num_represented_assemblies.append(result[0])
                    overlaps.append(result[1])
                    num_assemblies_per_branch.append(result[2])
                else:
                    num_represented_assemblies.append(result[0])
                    num_assemblies_per_branch.append(result[1])

            if make_plots:
                # Make some plots.
                _plot_weight_evolution_selected_branches(experiment,
                                                         input_directory,
                                                         plots_directory,
                                                         branches=branches,
                                                         patterns=patterns,
                                                         subsample=1, use_colors=True)

        mean_num_represented_assemblies.append(np.mean(num_represented_assemblies))
        std_num_represented_assemblies.append(np.std(num_represented_assemblies))

        if comp_stats:
            num_assemblies = _load_configuration(
                input_directories[0], experiment, ["input_parameters", "num_assemblies"])

            logfile.write("Represented assemblies {0:0.3f} "
                          "SD: {1:0.3f}/{2} {3}\n"
                          .format(mean_num_represented_assemblies[-1],
                                  std_num_represented_assemblies[-1],
                                  num_assemblies,
                                  num_represented_assemblies).encode())
            logfile.write("Mean num assemblies per branch {0:0.3f} "
                          "SD: {1:0.3f}\n"
                          .format(np.mean(num_assemblies_per_branch),
                                  np.std(num_assemblies_per_branch)).encode())
            mm = []
            for i in range(num_assemblies + 1):
                mm.append([list(num).count(i) for num in
                           num_assemblies_per_branch])
            m = np.mean(mm, axis=1)
            s = np.std(mm, axis=1)
            logfile.write("Num assemblies per branch\n"
                          "0:{0:0.3f} SD: {1:0.3f}\n1:{2:0.3f} SD: {3:0.3f}\n"
                          "2:{4:0.3f} SD: {5:0.3f}\n3:{6:0.3f} SD: {7:0.3f}\n"
                          "4:{8:0.3f} SD: {9:0.3f}\n"
                          "5:{10:0.3f} SD: {11:0.3f}\n"
                          "6:{12:0.3f} SD: {13:0.3f}\n"
                          "7:{14:0.3f} SD: {15:0.3f}\n"
                          "8:{16:0.3f} SD: {17:0.3f}\n"
                          .format(m[0], s[0], m[1], s[1], m[2], s[2], m[3],
                                  s[3], m[4], s[4], m[5], s[5], m[6], s[6],
                                  m[7], s[7], m[8], s[8]).encode())

    if (experiment == "rewiring_ex5" or experiment == "rewiring_ex6") and comp_stats:
        idc = np.triu_indices_from(overlaps[0], k=1)
        triu = [overlaps[i][idc] for i in range(len(overlaps))]
        logfile.write("Mean overlap {0:0.4f} SD: {1:0.4f}\n"
                      .format(np.mean(triu),
                              np.std(triu)).encode())

        logfile.close()

    if experiment == "rewiring_ex4" and len(paths) > 1 and comp_stats:
        a, m, s = zip(*sorted(zip(neuron_activation_probability,
                                  mean_num_represented_assemblies,
                                  std_num_represented_assemblies),
                              reverse=True))
        _plot_num_represented_assemblies_over_act_probability(a, m, s, plots_directory)


if __name__ == "__main__":
    import sys
    main(sys.argv[1], sys.argv[2])
