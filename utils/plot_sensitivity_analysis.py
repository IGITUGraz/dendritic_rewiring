#!/usr/bin/env python

import os
import subprocess

import plotting.configure_seaborn as cs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set(context='paper', style='ticks', rc=cs.rc_params)

plots_directory = os.path.join(os.path.sep, "calc", os.getenv("USER"),
                               "data", "dendritic_rewiring", "rewiring_ex2/sensitivity_analysis")

mean_best = 7.400
std_best = 0.566

# These values have to be set to the values obtained via `stats_rewiring.py`.
mean_assemblies_neg = [7.320,
                       7.080,
                       7.240,
                       7.280,
                       7.080,
                       7.440,
                       7.200,
                       6.360,
                       7.240,
                       7.280,
                       7.240,
                       7.400
                       ]
std_assemblies_neg = [0.676,
                      0.891,
                      0.650,
                      0.665,
                      0.744,
                      0.753,
                      0.748,
                      0.975,
                      0.709,
                      0.776,
                      0.763,
                      0.566
                      ]

mean_assemblies_pos = [7.400,
                       7.280,
                       7.200,
                       7.040,
                       7.080,
                       7.040,
                       7.160,
                       6.400,
                       7.600,
                       7.280,
                       7.360,
                       7.400
                       ]
std_assemblies_pos = [0.632,
                      0.722,
                      0.632,
                      0.720,
                      0.845,
                      0.720,
                      0.731,
                      0.849,
                      0.566,
                      0.665,
                      0.742,
                      0.566
                      ]

mean_assemblies = np.array([mean_assemblies_pos, mean_assemblies_neg])
std_assemblies = np.array([std_assemblies_pos, std_assemblies_neg])

diff_mean = np.abs(mean_best - mean_assemblies)
max_diff_mean = np.max(diff_mean, axis=0)
min_diff_mean = np.min(diff_mean, axis=0)
idc = np.argmax(diff_mean, axis=0)

# sdi = max_diff_mean / np.diag(std_assemblies[idc])
sdi = 100/10 * max_diff_mean / mean_best
# sdi = 100/10 * min_diff_mean / mean_best
# sdi = 100/10 * np.mean(diff_mean, axis=0) / mean_best

print(diff_mean)
print(sdi)

param = [r"$\eta$", r"$T$", r"$c_{\mathcal{L}}$", r"$\gamma$", r"$\lambda$", r"$c_{\mathrm{w}}$",
         r"$c_{\mathcal{STDP}}$", r"$\mathcal{STDP}_{\mathrm{th}}$", r"$\tau_{\mathrm{x}}$",
         r"$c_{\mathrm{ds}}$", r"$\Delta_{\mathrm{min}}^{\mathrm{ds}}$",
         r"$\Delta_{\mathrm{max}}^{\mathrm{ds}}$"]

df = pd.DataFrame(data={param[i]: sdi[i] for i in range(len(sdi))}, index=range(len(sdi)))

colors = sns.color_palette().as_hex()
cs.set_figure_size(176 + 7, 60 + 6)

fig = plt.figure()
ax = sns.barplot(data=df, ci=None, color=colors[0])
ax.set_ylim([0, 1.4])
ax.set_ylabel("$\mathrm{SI}\%$")
ax.set_xlabel("Parameter")

plt.tight_layout()
fname = os.path.join(plots_directory, "sensitivity")
fig.savefig(fname + ".pdf", pad_inches=0.01)
subprocess.call(["pdftops", "-eps", fname + ".pdf", fname + ".eps"])

plt.close(fig)
