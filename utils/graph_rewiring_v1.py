#!/usr/bin/env python

import os
import subprocess

import ruamel_yaml as yaml

import configure_seaborn as cs
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import seaborn as sns
import utils as utils
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Ellipse, FancyArrowPatch, Polygon

sns.set(context='paper', style='ticks', rc=cs.rc_params)


def load_configuration(filepath):
    with open(filepath, "r") as f:
        config = yaml.safe_load(f)

    return config


def draw_connections(G, pos, node_colors, ax, edge_weights=None):

    alpha = 1.0
    for n in G:
        c = Ellipse(pos[n], width=0.015, height=0.015,
                    alpha=alpha, color=node_colors[n], clip_on=False)
        ax.add_patch(c)
        G.nodes[n]["patch"] = c
        x, y = pos[n]
    seen = {}
    alpha = 1.0  # 0.8
    for (u, v, d) in G.edges(data=True):
        n1 = G.nodes[u]["patch"]
        n2 = G.nodes[v]["patch"]
        rad = 0.1
        if (u, v) in seen:
            rad = seen.get((u, v))
            rad = (rad + np.sign(rad) * 0.1) * -1
        color = node_colors[u]
        e = FancyArrowPatch(n1.center,
                            n2.center,
                            patchA=n1,
                            patchB=n2,
                            shrinkA=0,
                            shrinkB=0,
                            arrowstyle='-',
                            linewidth=0.5,
                            connectionstyle="arc3, rad=%s" % rad,
                            mutation_scale=10.0,
                            alpha=alpha,
                            color=color,
                            clip_on=False)
        ax.add_patch(e)
        seen[(u, v)] = rad


def draw_assemblies(G, assemblies, colors):
    node_colors = []
    for i, assembly in enumerate(assemblies):
        G.add_nodes_from(assembly)
        node_colors += [colors[i]] * len(assembly)

    return node_colors


def draw_neuron(G, ax, branch_nodes, center_assemblies):
    x_c = center_assemblies
    x_branch_end = [x_c - 0.75, x_c - 0.2,  x_c + 0.6]

    b1 = lambda x: -0.2666667 * x - 0.573 # noqa
    b2 = lambda x: -3.18 * x - 1.35 # noqa
    b3 = lambda x: 0.666667 * x - 0.423 # noqa
    b_fun = [b1, b2, b3]

    # Branch nodes
    pos_branch_nodes = []
    for xe, bf, branch_node in zip(x_branch_end, b_fun, branch_nodes):
        dx = (x_c - xe) / 4
        pos_branch_nodes += [(x_c - dx, bf(x_c - dx)),
                             (x_c - 2 * dx, bf(x_c - 2 * dx)),
                             (x_c - 3 * dx, bf(x_c - 3 * dx))]
        G.add_nodes_from(branch_node)
    node_colors = ["w"] * len(pos_branch_nodes)

    # Neuron
    xy = np.array([[x_c, -1.6], [x_c + 0.07, -1 + y_offset],
                   [x_c - 0.07, -1 + y_offset]])
    nrn = Polygon(xy, clip_on=False, fill=False, color="k", lw=1)
    ax.add_patch(nrn)

    trunk = mlines.Line2D([x_c, x_c], [-1.6, -0.5], clip_on=False, color="k", linewidth=1)
    ax.add_line(trunk)

    branch1 = mlines.Line2D([x_c - 0.01, x_c - 0.75], [-0.5, -0.3], clip_on=False, color="k", linewidth=1)
    ax.add_line(branch1)

    branch2 = mlines.Line2D([x_c - 0.003, x_c - 0.2], [-0.486, 0.15], clip_on=False, color="k", linewidth=1)
    ax.add_line(branch2)

    branch3 = mlines.Line2D([x_c + 0.01, x_c + 0.6], [-0.6, -0.2], clip_on=False, color="k", linewidth=1)
    ax.add_line(branch3)

    return pos_branch_nodes, node_colors


def add_connections(experiment, weights, assemblies, assembly_idc, assembly_map, idc_other_assemblies,
                    num_neurons_per_assembly, idc_branch_nodes, min_plot_weight):
    conn = []
    for i, w in enumerate(weights):
        nrns = np.where(w > min_plot_weight)[0]
        map_idx_a_idx_nb = {}
        last_idx_nb = 0
        for nrn in nrns:
            idx_a = np.random.choice(
                map_neuron_id_to_assembly_id(nrn, assembly_idc))
            if experiment == "rewiring_ex3":
                if idx_a in map_idx_a_idx_nb:
                    idx_nb = map_idx_a_idx_nb[idx_a]
                else:
                    if last_idx_nb == 0:
                        idx_nb = 2
                        last_idx_nb = 2
                    else:
                        idx_nb = 0
                        last_idx_nb = 0
                    map_idx_a_idx_nb[idx_a] = idx_nb
            else:
                idx_nb = np.random.choice(idc_branch_nodes)
            idx_nrn = np.random.choice(num_neurons_per_assembly)
            if idx_a not in assembly_map.keys():
                idx_a = np.random.choice(idc_other_assemblies)
                conn.append((assemblies[idx_a][idx_nrn], branch_nodes[i][idx_nb]))
            else:
                conn.append((assemblies[assembly_map[idx_a]][idx_nrn], branch_nodes[i][idx_nb]))
    G.add_edges_from(conn)


def plot_input_spikes(input_spike_times, input_size, ax, xlim, pattern_labels, pattern_colors):
    idc = np.where((input_spike_times[:, 0] >= xlim[0]) & (input_spike_times[:, 0] <= xlim[1]))

    ax.scatter(input_spike_times[idc, 0], input_spike_times[idc, 1], s=1.0, color="k", edgecolor="none")

    ax.set_xlim(xlim)
    ax.set_ylim([None, input_size + 20])
    ax.set_xticks([])
    ax.set_yticks([1, input_size])
    ax.set_yticklabels("%d" % f for f in ax.get_yticks())
    ax.set_ylabel(r"Input", labelpad=3.2)

    for p, (pl, pc) in enumerate(zip(pattern_labels, pattern_colors)):
        ax.plot([xlim[0] + p * 0.5 + 0.2, xlim[0] + (p + 1) * 0.5],
                [355, 355], color=pc, alpha=1.0, linestyle='-',
                linewidth=1, clip_on=False)
        ax.text(xlim[0] + (p + 1) * 0.5 - 0.15, 400, pl, ha="center", color=pc, alpha=1.0)


def plot_soma_potential(mem_soma, ax, xlim):
    idc = np.where((mem_soma[:, 0] >= xlim[0]) & (mem_soma[:, 0] <= xlim[1]))
    ax.plot(mem_soma[idc][:, 0], mem_soma[idc][:, 1], color="k", linewidth=0.6)

    ax.set_ylabel(r"$V^{\mathrm{soma}}$ [mV]")
    ax.set_xlabel(r"$t$ [s]")
    ax.set_ylim([None, -25])
    ax.set_yticks([-70, -25])
    ax.set_yticklabels("%d" % f for f in [-70, -25])
    ax.set_xlim(xlim)
    xticks = np.linspace(xlim[0], xlim[1], 3)
    ax.set_xticks(xticks)
    ax.set_xticklabels("%.1f" % (f - xlim[0]) for f in xticks)


def plot_branch_potential(mem_branch, branch_id, ax, xlim):
    idc = np.where((mem_branch[:, 0] >= xlim[0]) & (mem_branch[:, 0] <= xlim[1]))
    ax.plot(mem_branch[idc][:, 0] - 0 * xlim[0], mem_branch[idc][:, 1], color="k", linewidth=0.6)
    ax.set_xlim(xlim)
    ax.set_ylim([-72, -25])
    xticks = ax.get_xticks()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylabel(r"$V^{\mathrm{b}}_{" + str(branch_id) + "}$", rotation=0, va="center")
    if branch_id >= 10:
        ax.yaxis.set_label_coords(-0.044, 0.75)
    else:
        ax.yaxis.set_label_coords(-0.05, 0.75)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_visible(False)

    return xticks


def plot_scale(xlim):
    line = mlines.Line2D([xlim[0] + 0.006, xlim[0] + 0.306], [-73.5, -73.5], clip_on=False, color=".15",
                         linewidth=0.7)
    ax.add_line(line)
    ax.text(np.mean([xlim[0] + 0.006, xlim[0] + 0.306]), -86.5, r"0.3 s", ha="center", fontsize=8)

    line = mlines.Line2D([xlim[0] - 0.03, xlim[0] - 0.03], [-68.9, -48.9], clip_on=False, color=".15",
                         linewidth=0.7)
    ax.add_line(line)
    ax.text(xlim[0] - 0.365, -64, r"20 mV", fontsize=8)


def map_neuron_id(gid, old_min=0, old_max=319, new_min=0, new_max=35):
    old_range = (old_max - old_min)
    new_range = (new_max - new_min)
    return int(((((gid - old_min) * new_range) / old_range) + new_min))


def map_neuron_id_to_assembly_id(gid, assembly_idc):
    if not any(gid in x for x in assembly_idc):
        return [-1]

    return np.where(assembly_idc == gid)[0]


# ------------------------------------------------------------------------------
experiment = "rewiring_ex1"
sim_date = "191125_135334/1"
patterns = [0, 1, 4]
patterns_graph = [0, 1, 2]
branches = [1, 5, 11]
xlims = [[0.0, 1.7], [1.5, 3.2]]

cs.set_figure_size(84 + 9, 87 + 8)

# ------------------------------------------------------------------------------
# Directory of simulation results and log files.
if experiment == "rewiring_ex4":
    input_directory = os.path.join("results", experiment, "4", sim_date, "data")
else:
    input_directory = os.path.join("results", experiment, sim_date, "data")

# Directory for plots.
plots_directory = os.path.join(input_directory, "..", "plots")
if not os.path.exists(plots_directory):
    os.makedirs(plots_directory)

# Colors of patters.
c = sns.color_palette().as_hex()
c[4], c[5], c[6], c[7], c[8], c[9] = c[4], c[8], c[5], c[6], c[8], c[7]

colors = [c[9]]
colors += [c[i] for i, p in enumerate(patterns)]
colors += [c[9]]
pattern_labels = [[r"$\mathrm{A}_%d$" % (p + 1) for p in patterns],
                  [r"$\mathrm{A}_%d$" % (p + 1) for p in patterns]]
pattern_colors = [[c[p] for p in patterns], [c[p] for p in patterns]]
assembly_map = {p: i + 1 for i, p in enumerate(patterns_graph)}
branches.sort()

np.random.seed(0)
num_rows = 3
node_colors = []
num_branches = 3
num_assemblies = 5
min_plot_weight = 1.0
idc_other_assemblies = [0, 4]
num_assemblies_real = 3
num_branch_nodes = 3
num_neurons_per_row = 7
num_neurons_per_assembly = num_rows * num_neurons_per_row
pos_assemblies = []

x_offset = -0.54
y_offset = -0.79
for i in range(num_rows):
    for x in np.linspace(-1 + x_offset, 1, 35):
        pos_assemblies.append((x, 1.0 - i * 0.1))
np.random.shuffle(pos_assemblies)

assemblies = np.split(np.arange(num_neurons_per_assembly * num_assemblies), num_assemblies)

branch1_nodes = np.max(assemblies) + 1 + np.arange(num_branch_nodes)
branch2_nodes = np.max(branch1_nodes) + 1 + np.arange(num_branch_nodes)
branch3_nodes = np.max(branch2_nodes) + 1 + np.arange(num_branch_nodes)
branch_nodes = [branch1_nodes, branch2_nodes, branch3_nodes]

if experiment == "rewiring_ex5":
    idc_branch_nodes = [0, 2]
else:
    idc_branch_nodes = range(num_branch_nodes)

# Load the configuration file.
config = utils.load_configuration(os.path.join(
    input_directory, "..", "config_" + experiment + ".yaml"))
sim_simulation_time = config["simulation_time"]
sim_w_max = config["connection_parameters"]["w_max"]
sim_input_size = config["input_parameters"]["num_inputs"]
sim_num_assemblies = config["input_parameters"]["num_assemblies"]
sim_assembly_size = config["input_parameters"]["assembly_size"]
sim_num_branches = config["neuron_parameters"]["num_branches"]
sim_sampling_interval_weights = config["sampling_interval_weights"]
input_size = config["input_parameters"]["num_inputs"]

if experiment == "rewiring_ex5":
    assembly_idc = []
    assembly_idc = np.loadtxt(
        os.path.join(input_directory, "assembly_neurons_idc"), dtype=np.int)
    # for assembly_idx_low in np.sort(assembly_idc_low):
    #     assembly_idc.append(np.arange(assembly_idx_low, assembly_idx_low +
    #                                   sim_assembly_size))
else:
    assembly_idc = np.split(np.arange(sim_num_assemblies * sim_assembly_size),
                            sim_num_assemblies)

# Load the simulation results.
mem_branch = []
for b in branches:
    mem_branch.append(np.loadtxt(os.path.join(
        input_directory, "test_branch" + str(b) + ".0.mem")))
mem_soma = np.loadtxt(os.path.join(input_directory, 'test_soma.0.mem'))
input_spike_times = np.loadtxt(os.path.join(
    input_directory, 'test_input.0.ras'))
input_spike_times[:, 1] = np.random.randint(0, sim_input_size,
                                            len(input_spike_times))

header_lenght = 3
with open(os.path.join(input_directory, "weights.0.dat"), "rb") as f:
    lines = f.readlines()
weights_pre = np.loadtxt(lines[header_lenght:sim_num_branches + header_lenght])
weights_train_start = [weights_pre[b] for b in branches]
weights_post = np.loadtxt(lines[-sim_num_branches:])
weights_train_end = [weights_post[b] for b in branches]

# Before training.
# ------------------------------------------------------------------------------
fig = plt.figure()
gs = GridSpec(6, 2)

# Input spikes.
ax = plt.subplot(gs[0, :])
plot_input_spikes(input_spike_times, input_size, ax, xlims[0], pattern_labels[0], pattern_colors[0])

# Create graph.
node_colors = []
G = nx.MultiGraph()
ax = plt.subplot(gs[1:-1, 0])

# Draw the assemblies.
node_colors += draw_assemblies(G, assemblies, colors)

# Draw the neuron.
xc = np.mean(pos_assemblies, axis=0)[0]
pos_branch_nodes, nc = draw_neuron(G, ax, branch_nodes, xc)
node_colors += nc

# Add connections to graph.
add_connections(experiment, weights_train_start, assemblies, assembly_idc,
                assembly_map, idc_other_assemblies, num_neurons_per_assembly,
                idc_branch_nodes, min_plot_weight)

# Draw connections.
draw_connections(G, (pos_assemblies + pos_branch_nodes), node_colors, ax)

plt.axis('off')
ax.set_xlim([-1, 1])
ax.set_ylim([-1 + y_offset, 1])

# Branch 3 potential.
ax = plt.subplot(gs[2, 1:])
xticks = plot_branch_potential(mem_branch[2], branches[2] + 1, ax, xlims[0])

# Scale
plot_scale(xlims[0])

# Branch 2 potential.
ax = plt.subplot(gs[3, 1:])
xticks = plot_branch_potential(mem_branch[1], branches[1] + 1, ax, xlims[0])

# Branch 1 potential.
ax = plt.subplot(gs[4, 1:])
xticks = plot_branch_potential(mem_branch[0], branches[0] + 1, ax, xlims[0])

# Soma potential.
ax = plt.subplot(gs[-1, :])
plot_soma_potential(mem_soma, ax, xlims[0])

plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.4)

fname = os.path.join(plots_directory, "graph-before")
fig.savefig(fname + ".pdf", pad_inches=0.01)
subprocess.call(["pdftops", "-eps", fname + ".pdf", fname + ".eps"])

plt.close(fig)


# After training.
# ------------------------------------------------------------------------------
fig = plt.figure()
gs = GridSpec(6, 2)

# Input spikes.
ax = plt.subplot(gs[0, :])
plot_input_spikes(input_spike_times, input_size, ax, xlims[1], pattern_labels[1], pattern_colors[1])

# Create graph.
node_colors = []
G = nx.MultiGraph()
ax = plt.subplot(gs[1:-1, 0])

# Draw the assemblies.
node_colors += draw_assemblies(G, assemblies, colors)

# Draw the neuron.
xc = np.mean(pos_assemblies, axis=0)[0]
pos_branch_nodes, nc = draw_neuron(G, ax, branch_nodes, xc)
node_colors += nc

# Add connections to graph.
add_connections(experiment, weights_train_end, assemblies, assembly_idc,
                assembly_map, idc_other_assemblies, num_neurons_per_assembly,
                idc_branch_nodes, min_plot_weight)

# Draw connections.
draw_connections(G, (pos_assemblies + pos_branch_nodes), node_colors, ax)

plt.axis('off')
ax.set_xlim([-1, 1])
ax.set_ylim([-1 + y_offset, 1])

# Branch 3 potential.
ax = plt.subplot(gs[2, 1:])
xticks = plot_branch_potential(mem_branch[2], branches[2] + 1, ax, xlims[1])

# Scale
plot_scale(xlims[1])

# Branch 2 potential.
ax = plt.subplot(gs[3, 1:])
xticks = plot_branch_potential(mem_branch[1], branches[1] + 1, ax, xlims[1])

# Branch 1 potential.
ax = plt.subplot(gs[4, 1:])
xticks = plot_branch_potential(mem_branch[0], branches[0] + 1, ax, xlims[1])

# Soma potential.
ax = plt.subplot(gs[-1, :])
plot_soma_potential(mem_soma, ax, xlims[1])

plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.4)

fname = os.path.join(plots_directory, "graph-after")
fig.savefig(fname + ".pdf", pad_inches=0.01)
subprocess.call(["pdftops", "-eps", fname + ".pdf", fname + ".eps"])

plt.close(fig)
