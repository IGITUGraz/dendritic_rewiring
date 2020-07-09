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
TODO (sections correspond to the sections of the paper).
Raw simulation data will be stored in the folder [results](results).

### Synaptic clustering through rewiring
Here we show that synaptic rewiring dynamics give rise to (a) clustering of functionally related inputs onto dendritic compartments, and (b) segregation of different assemblies onto different branches. This leads to segregated assembly-specific activation of dendritic branches. To reproduce the raw results reportet in the paper, run `python sim_rewiring_ex1.py` (default parameters will be parsed from [config_rewiring_ex1.yaml](config_rewiring_ex1.yaml)).

### STDP increases the capacity of neurons to store assembly patterns
With the plasticity dynamics considered above, since plasticity depends solely on synaptic inputs and local dendritic potentials, all branches are adapted independently without taking the activity of other branches into account. This implies that synaptic patterns at different branches can become correlated in the sense that projections from one assembly cluster on two or more branches of the neuron. Here we show that a simple additional STDP mechanism indirectly introduces competition between dendritic branches. This increases the capacity of the neuron in terms of the number of assemblies that are stored on dendrites. To reproduce the raw results reportet in the paper, run `python sim_rewiring_ex2.py` (default parameters will be parsed from [config_rewiring_ex2.yaml](config_rewiring_ex2.yaml)).

TODO: Simulations with alternative plasticity rule

### Rewiring protects stored information
One effect of the synaptic clustering observed in our simulations was that activation of different input assemblies led to dendritic spikes in different dendritic branches. [Cichon and Gan (2015)](https://www.nature.com/articles/nature14251) argued that such segregation may provide a mechanism to shelter previous memories from being overwritten by novel plasticity events. In a pattern memorization task we have shown that synaptic clustering indeed supports memory protection. To reproduce the raw results reportet in the paper, run `python sim_rewiring_ex3.py` (default parameters will be parsed from [config_rewiring_ex3.yaml](config_rewiring_ex3.yaml)).

We hypothesized that competition between branches through STDP is essential for this capability. Due to the competition, only one or a few branches evolve a synaptic cluster from a specific assembly while all other branches remain neutral to assemblies. This is especially important when assemblies are presented in a sequential manner. Without STDP the mean number of represented assemblies significantly decrease. To reproduce this, run `python sim_rewiring_ex3.py` again but with `stdp_active` set to `False` in [config_rewiring_ex3.yaml](config_rewiring_ex3.yaml).

### Synaptic clustering depends on input statistics
In the above simulations, input assemblies were activated in temporal isolation and each assembly activation was clean in the sense that all assembly neurons had an elevated firing rate and assemblies were disjoint. Under these conditions, our rewiring model led to robust synaptic clustering. We next determined the sensitivity of this result on these input statistics.

*Influence of co-active assemblies:* We first asked whether input assemblies can be clustered onto different dendritic compartments even when single assemblies are never activated in temporal isolation. We have shown that our rewiring mechanisms segregates uncorrelated inputs onto different branches even when assembly patterns are not temporally isolated (the number of represented assemblis decreases although with increasing number of simultaneous active assemblies). To reproduce the raw results reportet in the paper, run `python sim_rewiring_ex4.py` (default parameters will be parsed from [config_rewiring_ex4.yaml](config_rewiring_ex4.yaml)). To set the number of simultaneous active assemblies, edit `num_simultaneous_assemblies` in [config_rewiring_ex4.yaml](config_rewiring_ex4.yaml) (note that we kept the average time that an assembly is active constant; to do so, set the `simulation_time` parameter in [config_rewiring_ex4.yaml](config_rewiring_ex4.yaml) to `1000.0/num_simultaneous_assemblies`).

*Influence of stochastic assembly activation:* Here we investigated how clustering in our model depends on the reliability of assembly activations. To this end, we repeated the experiment with simultaneously activated assemblies but with a reduced number of active neurons in each assembly activation. The number of assembly-synapse clusters on dendritic branches decreased with the neuron activation probability. To reproduce the raw results reportet in the paper, run `python sim_rewiring_ex4.py` with different values for `neuron_activation_probability` (see [config_rewiring_ex4.yaml](config_rewiring_ex4.yaml)).

*Influence of assembly overlap:* TODO

### Suplementary experiments
TODO

#### Analysis of the influence of dendritic spikes on synaptic clustering
TODO

#### Sensitivity analysis of parameters
TODO

#### Analysis of the influence of pattern duration and background interval on clustering
TODO

#### Analysis of the weakening of older memories over time
TODO

#### Retaining older memories for an extended period of time
TODO

#### Synaptic consolidation by sharpening of weight priors
TODO

## References

* Limbacher, T., & Legenstein, R. (2020). Emergence of Stable Synaptic Clusters on Dendrites Through Synaptic Rewiring. Frontiers in Computational Neuroscience, 14, 57.
