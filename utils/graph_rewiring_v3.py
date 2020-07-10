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
        c = Ellipse(pos[n], width=0.013, height=0.013,
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
                            connectionstyle="arc3, rad=%s" % rad,
                            mutation_scale=10.0,
                            alpha=alpha,
                            lw=0.5,
                            color=color,
                            clip_on=False)
        ax.add_patch(e)
        seen[(u, v)] = rad


def draw_assemblies(G, assemblies, colors):
    node_colors = []
    for i, assembly in enumerate(assemblies):
        G.add_nodes_from(assembly)
    for i in range(len(assemblies)):
        if i == 0:
            node_colors += [colors[i]] * 14
        else:
            node_colors += [colors[i]] * 13

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
    xy = np.array([[x_c, -1.6], [x_c + 0.07, -1 + y_offset], [x_c - 0.07, -1 + y_offset]])
    nrn = Polygon(xy, clip_on=False, fill=False, color="k", lw=1)
    ax.add_patch(nrn)

    # Trunk
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
    idc_a = []
    ii = []
    for i, w in enumerate(weights):
        nrns = np.where(w > min_plot_weight)[0]
        print(len(nrns))
        for nrn in nrns:
            idc_a.append(np.random.choice(map_neuron_id_to_assembly_id(nrn, assembly_idc)))
            ii.append(i)
    for idx_a, i in zip(idc_a, ii):
        idx_nb = np.random.choice(idc_branch_nodes)
        idx_nrn = np.random.choice(num_neurons_per_assembly[idx_a])
        if idx_a not in assembly_map.keys():
            idx_a = np.random.choice(idc_other_assemblies)
            conn.append((assemblies[idx_a][idx_nrn], branch_nodes[i][idx_nb]))
        else:
            conn.append((assemblies[assembly_map[idx_a]][idx_nrn], branch_nodes[i][idx_nb]))
    G.add_edges_from(conn)


def plot_soma_potential(mem_soma, ax, xlim, pattern_labels, pattern_colors):
    idc = np.where((mem_soma[:, 0] >= xlim[0]) & (mem_soma[:, 0] <= xlim[1]))
    ax.plot(mem_soma[idc][:, 0], mem_soma[idc][:, 1], color="k",
            clip_on=False, linewidth=0.6)

    for p, (pl, pc) in enumerate(zip(pattern_labels, pattern_colors)):
        ax.plot([xlim[0] + p * 0.5 + 0.2, xlim[0] + (p + 1) * 0.5], [-12, -12], color=pc, alpha=1.0,
                linestyle='-', linewidth=0.7, clip_on=False)
        ax.text(xlim[0] + (p + 1) * 0.5 - 0.15, -5, pl, ha="center", color=pc, alpha=1.0, fontsize=8)

    ax.set_ylabel(r"$V^{\mathrm{soma}}$", rotation=0, va="center")
    ax.yaxis.set_label_coords(-0.1, 1.4)
    # ax.set_ylim([None, 20])
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_xlim(xlim)

    for spine in ["top", "right", "left", "bottom"]:
        ax.spines[spine].set_visible(False)


def plot_scale(xlim):
    line = mlines.Line2D([xlim[0] + 0.006, xlim[0] + 0.306], [-79.0, -79.0], clip_on=False, color="0.15",
                         linewidth=0.7)
    ax.add_line(line)
    ax.text(np.mean([xlim[0] + 0.006, xlim[0] + 0.306]), -106, r"0.3 s", ha="center", fontsize=8)

    line = mlines.Line2D([xlim[0] - 0.04, xlim[0] - 0.04], [-68.9, -43], clip_on=False, color="0.15",
                         linewidth=0.7)
    ax.add_line(line)
    ax.text(xlim[0] - 0.44, -66, r"25 mV", fontsize=8)


def map_neuron_id(gid, old_min=0, old_max=319, new_min=0, new_max=35):
    old_range = (old_max - old_min)
    new_range = (new_max - new_min)
    return int(((((gid - old_min) * new_range) / old_range) + new_min))


def map_neuron_id_to_assembly_id(gid, assembly_idc):
    if not any(gid in x for x in assembly_idc):
        return [-1]

    return np.where(assembly_idc == gid)[0]


# ------------------------------------------------------------------------------
experiment = "rewiring_ex3"
sim_date = "191204_132602/17"
branches = [0, 11, 10]
patterns = [0, 3, 6]
master_seed = 5
xlims = [[0.0, 1.7], [1.5, 3.2], [3.0, 4.7]]

# t = 1000s
# idc1 = 5424
# idc2 = 21674
# idc3 = 43345
# t = 2000s
idc1 = 10845
idc2 = 43345
idc3 = 75845
# t = 5000s
# idc1 = 27095
# idc2 = 108345
# idc3 = 189595


cs.set_figure_size(58 + 6, 45 + 8)

# ------------------------------------------------------------------------------
# Directory of simulation results and log files.
input_directory = os.path.join("results", experiment, sim_date, "data")

# Directory for plots.
plots_directory = os.path.join(input_directory, "..", "plots")
if not os.path.exists(plots_directory):
    os.makedirs(plots_directory)

np.random.seed(master_seed)
num_rows = 3
node_colors = []
num_branches = 3
num_assemblies = 8
min_plot_weight = 1.3
num_assemblies_real = 8
num_branch_nodes = 3
num_neurons_per_row = 7
num_neurons_per_assembly = [14]
num_neurons_per_assembly += 7 * [13]
pos_assemblies = []

idc_other_assemblies1 = list(range(8))
idc_other_assemblies2 = list(range(8))
idc_other_assemblies3 = list(range(8))
idc_other_assemblies1.remove(patterns[0])
idc_other_assemblies2.remove(patterns[1])
idc_other_assemblies3.remove(patterns[2])

# Colors of patters.
# c = plt.rcParams['axes.prop_cycle'].by_key()['color']
# c[6] = c[8]
# c[7] = c[9]
c = sns.color_palette().as_hex()
c[4], c[5], c[6], c[7], c[8], c[9] = c[4], c[8], c[5], c[6], c[8], c[9]

pattern_labels = [[r"$\mathrm{A}_%d$" % (p + 1) for p in patterns],
                  [r"$\mathrm{A}_%d$" % (p + 1) for p in patterns]]
pattern_colors = [[c[p] for p in patterns], [c[p] for p in patterns]]
assembly_map = {p: p for p in patterns}
branches.sort()

x_offset = -0.54
y_offset = -0.79
for i in range(num_rows):
    for x in np.linspace(-1 + x_offset, 1, 35):
        pos_assemblies.append((x, 1.0 - i * 0.1))
np.random.shuffle(pos_assemblies)

assemblies = []
for i in range(num_assemblies):
    if i == 0:
        assemblies.append(np.arange(14))
    else:
        assemblies.append(np.max(assemblies[-1]) + 1 + np.arange(13))

branch1_nodes = np.max(assemblies[-1]) + 1 + np.arange(num_branch_nodes)
branch2_nodes = np.max(branch1_nodes) + 1 + np.arange(num_branch_nodes)
branch3_nodes = np.max(branch2_nodes) + 1 + np.arange(num_branch_nodes)
branch_nodes = [branch1_nodes, branch2_nodes, branch3_nodes]

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

assembly_idc = np.split(np.arange(sim_num_assemblies * sim_assembly_size),
                        sim_num_assemblies)

# Load the simulation results.
mem_soma = np.loadtxt(os.path.join(input_directory, 'test_soma.0.mem'))

with open(os.path.join(input_directory, "weights.0.dat"), "rb") as f:
    lines = f.readlines()

weights = np.loadtxt(lines[idc1:idc1 + sim_num_branches])
weights1 = [weights[b] for b in branches]
weights = np.loadtxt(lines[idc2:idc2 + sim_num_branches])
weights2 = [weights[b] for b in branches]
weights = np.loadtxt(lines[idc3:idc3 + sim_num_branches])
weights3 = [weights[b] for b in branches]

# After assembly 1.
# ------------------------------------------------------------------------------
np.random.seed(master_seed)
fig = plt.figure()
gs = GridSpec(5, 2)

# Create graph.
node_colors = []
G = nx.MultiGraph()
ax = plt.subplot(gs[0:, 0])

# Draw the assemblies.
node_colors += draw_assemblies(G, assemblies, c)

# Draw the neuron.
xc = np.mean(pos_assemblies, axis=0)[0]
pos_branch_nodes, nc = draw_neuron(G, ax, branch_nodes, xc)
node_colors += nc

# Add connections to graph.
add_connections(experiment, weights1, assemblies, assembly_idc,
                assembly_map, idc_other_assemblies1, num_neurons_per_assembly,
                idc_branch_nodes, min_plot_weight)

# Draw connections.
draw_connections(G, (pos_assemblies + pos_branch_nodes), node_colors, ax)

plt.axis('off')
ax.set_xlim([-1, 1])
ax.set_ylim([-1 + y_offset, 1])

# Soma potential.
ax = plt.subplot(gs[-1, 1:])
plot_soma_potential(mem_soma, ax, xlims[0], pattern_labels[0],
                    pattern_colors[0])

# Scale
plot_scale(xlims[0])

plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                    wspace=-0.2, hspace=0.4)

fname = os.path.join(plots_directory, "graph1")
fig.savefig(fname + ".pdf", pad_inches=0.01)
subprocess.call(["pdftops", "-eps", fname + ".pdf", fname + ".eps"])
plt.close(fig)


# After assembly 2.
# ------------------------------------------------------------------------------
np.random.seed(master_seed)
fig = plt.figure()
gs = GridSpec(5, 2)

# Create graph.
node_colors = []
G = nx.MultiGraph()
ax = plt.subplot(gs[0:, 0])

# Draw the assemblies.
node_colors += draw_assemblies(G, assemblies, c)

# Draw the neuron.
xc = np.mean(pos_assemblies, axis=0)[0]
pos_branch_nodes, nc = draw_neuron(G, ax, branch_nodes, xc)
node_colors += nc

# Add connections to graph.
add_connections(experiment, weights2, assemblies, assembly_idc,
                assembly_map, idc_other_assemblies2, num_neurons_per_assembly,
                idc_branch_nodes, min_plot_weight)

# Draw connections.
draw_connections(G, (pos_assemblies + pos_branch_nodes), node_colors, ax)

plt.axis('off')
ax.set_xlim([-1, 1])
ax.set_ylim([-1 + y_offset, 1])

# Soma potential.
ax = plt.subplot(gs[-1, 1:])
plot_soma_potential(mem_soma, ax, xlims[1], pattern_labels[0],
                    pattern_colors[0])

# Scale
plot_scale(xlims[1])

plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                    wspace=-0.2, hspace=0.4)

fname = os.path.join(plots_directory, "graph2")
fig.savefig(fname + ".pdf", pad_inches=0.01)
subprocess.call(["pdftops", "-eps", fname + ".pdf", fname + ".eps"])
plt.close(fig)

# After assembly 3.
# ------------------------------------------------------------------------------
np.random.seed(master_seed)
fig = plt.figure()
gs = GridSpec(5, 2)

# Create graph.
node_colors = []
G = nx.MultiGraph()
ax = plt.subplot(gs[0:, 0])

# Draw the assemblies.
node_colors += draw_assemblies(G, assemblies, c)

# Draw the neuron.
xc = np.mean(pos_assemblies, axis=0)[0]
pos_branch_nodes, nc = draw_neuron(G, ax, branch_nodes, xc)
node_colors += nc

# Add connections to graph.
add_connections(experiment, weights3, assemblies, assembly_idc,
                assembly_map, idc_other_assemblies3, num_neurons_per_assembly,
                idc_branch_nodes, min_plot_weight)

# Draw connections.
draw_connections(G, (pos_assemblies + pos_branch_nodes), node_colors, ax)

plt.axis('off')
ax.set_xlim([-1, 1])
ax.set_ylim([-1 + y_offset, 1])

# Soma potential.
ax = plt.subplot(gs[-1, 1:])
plot_soma_potential(mem_soma, ax, xlims[2], pattern_labels[0],
                    pattern_colors[0])

# Scale
plot_scale(xlims[2])

plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                    wspace=-0.2, hspace=0.4)

fname = os.path.join(plots_directory, "graph3")
fig.savefig(fname + ".pdf", pad_inches=0.01)
subprocess.call(["pdftops", "-eps", fname + ".pdf", fname + ".eps"])
plt.close(fig)
