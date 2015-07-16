ssa
===

An implementation of Gillespie's stochastic simulation algorithm (SSA) with 1D diffusion

Dependencies
====

The easiest way to ensure all dependencies are met is to use a Python distribution, such as Anaconda (https://store.continuum.io/cshop/anaconda/) or Canopy (https://www.enthought.com/products/canopy/)

Specifically, this code uses:

NumPy (1.9)

Matplotlib (1.4)

PyYaml (http://pyyaml.org)

Cython (http://cython.org)


Installation
====

After cloning, simply run:
'''
./setup.sh
'''

Make sure you add `[ssa]/lib` and `[ssa]/src` to your `PYTHONPATH` environmental variable, where
'[ssa]' points to the directory of the ssa code.

You may also want to add `ssa/bin` to your `PATH`

Simulation setup
====

Simulations require two (YAML formatted) configuration files:

System configuration file

Defines and system and simulation run parameters:

* Run parameters (Run time, etc)
* Species definitions
* Reaction definitions
* Number of diffusion bins

Parameter configuration file

Defines parameters for the system defined in the system config file:

* Diffusion constants for each species
* Reaction rate constants
* initial species concentrations in each bin


See examples `examples` for more details:

*sim.cfg* - Simulation config

*params.cfg* - Parameter config


Running a simulation
=====

**ssa_init** : Preprocess configuration files and initialize simulation

**ssa_run** : Run or continue simulation from state file

You can use either command with option '-h' or '--help' for more information
