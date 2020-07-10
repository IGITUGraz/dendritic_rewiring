# Dendritic rewiring
This is the simulation code used in the paper "[Emergence of Stable Synaptic Clusters on Dendrites Through Synaptic
Rewiring ](https://www.frontiersin.org/articles/10.3389/fncom.2020.00057)".

![Schema neuron model and plasticity/rewiring](https://i.imgur.com/qezv4Z6.png)

## Setup
You need Python to run this code. We tested it on Python version 3.7. Additional dependencies are listed in
[environment.yml](environment.yml). If you use [Conda](https://docs.conda.io/en/latest/), run

```bash
conda env create --file=environment.yml
```

to install the required packages and their dependencies.

## Usage
In the following we show how to reproduce the results reported in the paper. There is a Python script and the corresponding configuration file for each experiment (see sections below; sections correspond to the sections of the paper). Raw simulation data will be stored in the folder [results](results).

Use [stats_rewiring.py](utils/stats_rewiring.py) to compute the number of represented assemblies, number of synapses per branch, etc. With this script you can also plot the evolution of the synaptic weights for selected branches (as in e.g., Figure 2). To do so, run the following after running the simulation script:

```bash
python utils/stats_rewiring.py "rewiring_ex1" "200710_140856/0"
```

where the first parameter is the simulation you want to analyze (e.g., rewiring_ex1, see below) and the second parameter is the simulation timestep (before /) and the trial ID (after /). This will create the file `results/rewiring_ex1/200710_140856/stats.txt` containing the stats.

If you want to plot the evolution of the synaptic weights over time, set the variable `make_plots` in [stats_rewiring.py](utils/stats_rewiring.py) to `True`.

### Synaptic clustering through rewiring
Here we show that synaptic rewiring dynamics give rise to (a) clustering of functionally related inputs onto dendritic compartments, and (b) segregation of different assemblies onto different branches. This leads to segregated assembly-specific activation of dendritic branches. To reproduce the raw results reportet in the paper, run `python sim_rewiring_ex1.py` (default parameters will be parsed from [config_rewiring_ex1.yaml](config_rewiring_ex1.yaml)). To plot the wiring diagram before and after rewiring (as in Figure 2) run [eval_rewiring.py](utils/eval_rewiring.py) and then run [graph_rewiring_ex1.py](utils/graph_rewiring_ex1.py).

### STDP increases the capacity of neurons to store assembly patterns
With the plasticity dynamics considered above, since plasticity depends solely on synaptic inputs and local dendritic potentials, all branches are adapted independently without taking the activity of other branches into account. This implies that synaptic patterns at different branches can become correlated in the sense that projections from one assembly cluster on two or more branches of the neuron. Here we show that a simple additional STDP mechanism indirectly introduces competition between dendritic branches. This increases the capacity of the neuron in terms of the number of assemblies that are stored on dendrites. To reproduce the raw results reportet in the paper, run `python sim_rewiring_ex2.py` (default parameters will be parsed from [config_rewiring_ex2.yaml](config_rewiring_ex2.yaml)). To plot the wiring diagram (as in Figure 3) run [eval_rewiring.py](utils/eval_rewiring.py) and then run [graph_rewiring_ex2.py](utils/graph_rewiring_ex2.py).

To test whether our model is sensitive to alterations in the plasticity rules, we reran the simulations with an alternative plasticity rule (see *Materials and Methods* for details). We found that the model behaved comparably. To reproduce the raw results reportet in the paper, run `python sim_rewiring_ex7.py` (default parameters will be parsed from [config_rewiring_ex7.yaml](config_rewiring_ex7.yaml)).

### Rewiring protects stored information
One effect of the synaptic clustering observed in our simulations was that activation of different input assemblies led to dendritic spikes in different dendritic branches. [Cichon and Gan (2015)](https://www.nature.com/articles/nature14251) argued that such segregation may provide a mechanism to shelter previous memories from being overwritten by novel plasticity events. In a pattern memorization task we have shown that synaptic clustering indeed supports memory protection. To reproduce the raw results reportet in the paper, run `python sim_rewiring_ex3.py` (default parameters will be parsed from [config_rewiring_ex3.yaml](config_rewiring_ex3.yaml)). To plot the wiring diagram (as in Figure 4) run [eval_rewiring_sequential.py](utils/eval_rewiring_sequential.py) and then run [graph_rewiring_ex3.py](utils/graph_rewiring_ex3.py).

We hypothesized that competition between branches through STDP is essential for this capability. Due to the competition, only one or a few branches evolve a synaptic cluster from a specific assembly while all other branches remain neutral to assemblies. This is especially important when assemblies are presented in a sequential manner. Without STDP the mean number of represented assemblies significantly decreases. To reproduce this, run `python sim_rewiring_ex3.py` again but with `stdp_active` set to `False` in [config_rewiring_ex3.yaml](config_rewiring_ex3.yaml).

### Synaptic clustering depends on input statistics
In the above simulations, input assemblies were activated in temporal isolation and each assembly activation was clean in the sense that all assembly neurons had an elevated firing rate and assemblies were disjoint. Under these conditions, our rewiring model led to robust synaptic clustering. We next determined the sensitivity of this result on these input statistics.

*Influence of co-active assemblies:* We first asked whether input assemblies can be clustered onto different dendritic compartments even when single assemblies are never activated in temporal isolation. Indeed, we have shown that our rewiring mechanisms segregates uncorrelated inputs onto different branches even when assembly patterns are not temporally isolated (the number of represented assemblies decreases although with increasing number of simultaneous active assemblies). To reproduce the raw results reportet in the paper, run `python sim_rewiring_ex4.py` (default parameters will be parsed from [config_rewiring_ex4.yaml](config_rewiring_ex4.yaml)). To set the number of simultaneous active assemblies, edit `num_simultaneous_assemblies` in [config_rewiring_ex4.yaml](config_rewiring_ex4.yaml) (note that we kept the average time that an assembly is active constant; to do so, set the `simulation_time` parameter in [config_rewiring_ex4.yaml](config_rewiring_ex4.yaml) to `1000.0/num_simultaneous_assemblies`).

*Influence of stochastic assembly activation:* Here we investigated how clustering in our model depends on the reliability of assembly activations. To this end, we repeated the experiment with simultaneously activated assemblies but with a reduced number of active neurons in each assembly activation. The number of assembly-synapse clusters on dendritic branches decreased with the neuron activation probability. To reproduce the raw results reportet in the paper, run `python sim_rewiring_ex4.py` with different values for `neuron_activation_probability` (see [config_rewiring_ex4.yaml](config_rewiring_ex4.yaml)).

*Influence of assembly overlap:* In the simulations so far, we considered disjoint input assemblies. To test whether synaptic clustering emerges also for input assemblies with various amounts of neuron overlap, we considered a setup where some of the neurons of each assembly were chosen from a shared pool of neurons. Our results show that synapses are organized on dendrites in a clustered manner even for assemblies that share a significant portion of neurons, where the number of represented assemblies decreases slightly with increasing assembly overlap. To reproduce the raw results reportet in the paper, run `python sim_rewiring_ex5.py` with different values for `num_assembly_neurons` (if you increase `num_assembly_neurons`, the assembly overlap increases; see the paper and [config_rewiring_ex5.yaml](config_rewiring_ex5.yaml) for more details).

In section *Rewiring protects stored information* we showed that our rewiring rule can support memory protection by recruiting branches sequentially as new assemblies were activated. To test whether this ability of the model depends on disjoint input assemblies, we repeated the experiment described in there but with overlapping assemblies and various assembly overlaps. Our model predicts that associated memories with overlapping assembly representations are segregated in an assembly-specific manner onto different dendritic branches and that this segregation can indeed support memory protection. To reproduce the raw results reportet in the paper, run `python sim_rewiring_ex6.py` with different values for `num_assembly_neurons` (if you increase `num_assembly_neurons`, the assembly overlap increases; see the paper and [config_rewiring_ex6.yaml](config_rewiring_ex6.yaml) for more details).

### Suplementary experiments
In the following we describe how to reproduce the results reportet in the supplementary.

#### Analysis of the influence of dendritic spikes on synaptic clustering
In our model we considered dendritic branch dynamics including a stochastic firing threshold and dendritic spikes where the shape of the dendritic spike is given by a brief sodium spikelet followed by a plateau. We hypothesized that this nonlinear integration of synaptic input and dendritic plateau potentials are crucial for synaptic clustering in our model. In order to verify this, we conducted simulations of an altered model with linear dendritic integration (i.e., we removed the firing threshold and did not model dendritic spikes). Our simulations indicate that the extended depolarization of dendritic plateau potentials is necessary to collectively strengthen correlated inputs which eventually leads to synaptic clustering. To reproduce the raw results reportet in the paper, run `python sim_rewiring_ex8.py` (default parameters will be parsed from [config_rewiring_ex8.yaml](config_rewiring_ex8.yaml)). To plot the wiring diagram before and after rewiring run [eval_rewiring.py](utils/eval_rewiring.py) and then run [graph_rewiring_ex1.py](utils/graph_rewiring_ex1.py).

#### Sensitivity analysis of parameters
We performed a one-at-a-time sensitivity analysis in order to investigate the impact of all plasticity parameters as well as three neuron parameters on the simulation results. Our analysis showed that the model is quite robust to parameter variations. To reproduce the raw results reportet in the paper, run `python sensitivity_analysis_ex2.py` (default parameters will be parsed from [config_rewiring_ex2.yaml](config_rewiring_ex2.yaml)). To run the sensitivity analysis for a specific parameter and the defiation from its default value, set `param` and `delta_param`, respectively. To plot the results use [plot_sensitivity_analysis.py](utils/plot_sensitivity_analysis.py).

#### Analysis of the influence of pattern duration and background interval on clustering
In the simulations so far, we considered a pattern duration of 300 ms and a background interval of 200 ms, and we presented a total of 2000 patterns in each experiment. Here we analyze the influence of the pattern duration and the duration of the background interval, that is the delay between successive patterns, on the model’s performance (measured by the number of assemblies represented on the neuron). We found that our rewiring mechanism is, up to some point, quite robust to variations in the pattern duration and the delay between patterns. To reproduce the raw results reportet in the paper, run `python sim_rewiring_ex9.py` (default parameters will be parsed from [config_rewiring_ex9.yaml](config_rewiring_ex9.yaml)).

#### Analysis of the weakening of older memories over time
In Section *Synaptic clustering depends on input statistics* we showed that rewiring supports memory protection by recruiting banches sequentially to store input patterns. Due to the random fluctuations of the weights that are imposed by the noise term in our rewiring dynamics, synaptic connections to inactive assemblies gradually degrade. This behavior can be interpreted as gradual forgetting. To analyze this weakening of older memories over time we recorded the synaptic weights of a branch that had a synaptic cluster to an inactive assembly and analyzed the average weight decrease per time unit for different strengths of the noise term. To reproduce the raw results reportet in the paper, run `python sim_rewiring_ex10.py` (default parameters will be parsed from [config_rewiring_ex10.yaml](config_rewiring_ex10.yaml)).

#### Retaining older memories for an extended period of time
Here we show here that the lifetime of older memories can be increased by using an additional Gaussian prior per synapse. To reproduce the raw results reportet in the paper, run `python sim_rewiring_ex11.py` with `gaussian_prior_thr` in [config_rewiring_ex11.yaml](config_rewiring_ex11.yaml) set to a value higher than `w_max` (e.g., 9.0; this effectively disables the sharpening of the prior, see below).

#### Synaptic consolidation by sharpening of weight priors
Above we described a way to reduce gradual forgetting in our model by introducing an additional term (that resulted from a Gaussian prior) in the rewiring dynamics. By using this additional Gaussian prior the stochastic plasticity dynamics tend to sample network configurations where the weight of a functional synapse is close to the mean of the Gaussian. Here, we investigate a simple way how consolidation can be incorporated in this framework by sharpening
of the Gaussian prior. To reproduce the raw results reportet in the paper, run `python sim_rewiring_ex11.py` (default parameters will be parsed from [config_rewiring_ex11.yaml](config_rewiring_ex11.yaml); use the default value for `gaussian_prior_thr`, i.e., 7.0).

## References

* Limbacher, T., & Legenstein, R. (2020). Emergence of Stable Synaptic Clusters on Dendrites Through Synaptic Rewiring. Frontiers in Computational Neuroscience, 14, 57.
