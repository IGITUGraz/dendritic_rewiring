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
Here we show that synaptic rewiring dynamics give rise to (a) clustering of functionally related inputs onto dendritic compartments, and (b) segregation of different assemblies onto different branches. This leads to segregated assembly-specific activation of dendritic branches. To reproduce the raw results reportet in the paper, run `python sim_rewiring_ex1.py` (default parameters will be parsed from [config_rewiring_ex1.yaml](config_rewiring_ex1.yaml).

### STDP increases the capacity of neurons to store assembly patterns
With the plasticity dynamics considered above, since plasticity depends solely on synaptic inputs and local dendritic potentials, all branches are adapted independently without taking the activity of other branches into account. This implies that synaptic patterns at different branches can become correlated in the sense that projections from one assembly cluster on two or more branches of the neuron. Here we show that a simple additional STDP mechanism indirectly introduces competition between dendritic branches. This increases the capacity of the neuron in terms of the number of assemblies that are stored on dendrites. To reproduce the raw results reportet in the paper, run `python sim_rewiring_ex2.py` (default parameters will be parsed from [config_rewiring_ex2.yaml](config_rewiring_ex2.yaml).

TODO: Simulations with alternative plasticity rule

### Rewiring protects stored information
One effect of the synaptic clustering observed in our simulations was that activation of different input assemblies led to dendritic spikes in different dendritic branches. 

### Synaptic clustering depends on input statistics
TODO

*Influence of co-active assemblies:* TODO

*Influence of stochastic assembly activation:* TODO

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
TODO
