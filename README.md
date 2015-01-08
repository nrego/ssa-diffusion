ssa
===

An implementation of Gillespie's stochastic simulation algorithm (SSA) with 1D diffusion

Dependencies
====

The easiest way to ensure all dependencies are met is to use the free Anaconda python distribution:

https://store.continuum.io/cshop/anaconda/

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

Running
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
