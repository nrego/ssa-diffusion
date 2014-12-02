'''
Created on Nov 26, 2014

@author: Nick Rego
'''
from __future__ import division, print_function; __metaclass__ = type

import numpy

from propensity import Propensity

import logging
log = logging.getLogger('system')


class System:
    '''
    Class representing simulation system parameters and state

    Initialized from configuration file with ssa_init

    Can output state as pickled state file

    Simulations can be ran/continued from state file with ssa_run
    '''

    def process_config(self):
        config = self.rc.config

        # Get system specification from config
        namespace = {'numpy': numpy,
                     'inf': float('inf')}

        try:
            compartments = eval(config.get(['system', 'spec', 'compartments'], '[0.0, 1.0]'), namespace)
        except Exception as e:
            raise ValueError('invalid compartment specification: {!r}'.format(e))

        self.compartment_bounds = numpy.array(list(compartments))
        self.compartment_lengths = numpy.diff(self.compartment_bounds)
        self.species = config.get(['system', 'spec', 'species'], [])
        self.species.sort()
        self.reactions = config.get(['system', 'spec', 'reactions'], {})

    def __init__(self, rc=None):
        self.rc = rc or ssa.rc

        # system spec stuff
        self.compartment_bounds = None
        self.compartment_lengths = None
        self.species = None
        self.reactions = None
        self.process_config()

        # system state
        self.state = None
        self._propensity = None

    # Initialize state from config - only during 'ssa_init'
    def initialize(self, paramfile):
        config = self.rc.config
        config.update_from_file(paramfile)
        log.info('Succesfully read in params file {}'.format(paramfile))
        log.debug('Updated config: {!r}'.format(config))

        self.rc.pstatus('Initializing system ...')

        # Initialize rates
        rates = {
            'reaction': {reaction: 0.0 for reaction in self.reactions.keys()},
            'diffusion': {specie: [0.0 for i in xrange(self.n_compartments)] for specie in self.species}
        }

        rate_params = config.get(['params', 'reaction'])
        diff_params = config.get(['params', 'diffusion'])

        if rate_params:
            rates['reaction'].update(rate_params)

        # Compartment diffusion rate: (d = D/h^2)
        # (D is diff coef, h is compartment length)
        if diff_params:
            squared_dist = numpy.power(self.compartment_lengths, 2)
            for specie in diff_params.keys() and self.species:
                diff_rates = diff_params[specie]/squared_dist
                rates['diffusion'][specie] = diff_rates

        # Initialize 'state' - i.e. what changes as simulation runs
        self.state = {'time': 0.0,
                      'n_species': {specie: [0 for i in self.compartment_lengths]
                                    for specie in self.species},
                      'rates': rates
                      }

        # System state from config file
        for specie, n_arr in self.n_species.items():
            conf_arr = config.get(['params', 'n_species', specie])
            if conf_arr:
                for i, val in enumerate(conf_arr):
                    if type(val) == int:
                        n_arr[i] = val

            self.state['n_species'][specie] = numpy.array(n_arr, dtype=numpy.uint32)

        self.rc.pstatus('...Done\n')
        self.rc.pflush()

        self.print_state()

    # dump current state info
    def print_state(self):

        self.rc.pstatus('Species:')
        for specie, n in self.n_species.items():
            self.rc.pstatus('  {!r}: {!r}'.format(specie, n))

        self.rc.pstatus('Compartments:')
        self.rc.pstatus('  N compartments: {!r}'.format(self.n_compartments))
        self.rc.pstatus('  Compartment bounds: {!r}'.format(self.compartment_bounds))
        self.rc.pstatus('  Compartment lengths: {!r}'.format(self.compartment_lengths))

        self.rc.pstatus('Rates:')
        self.rc.pstatus('  Species diffusion:')
        for specie, diffusion in self.diffusion_rates.items():
            self.rc.pstatus('    {!r}: d = {!r}'.format(specie, diffusion))
        self.rc.pstatus('  Reactions:')
        for rxn, rate in self.rxn_rates.items():
            self.rc.pstatus('    {!r}: {!r}'.format(rxn, rate))
        self.rc.pstatus()
        self.rc.pflush()

    # Simply reset state - for loading from cpt files, for example
    def load_state(self, state):
        self.state = state
        log.debug('Loaded state:{!r}'.format(state))
        #self.printState()

    def update_state_from_propensity(self):
        if self.propensity:
            for i, specie in enumerate(self.propensity.species):
                self.state['n_species'][specie] = self.propensity.n_species[i]

    # Run iteration - Note that rand is r2*a, where r2 element of [0, 1)
    def run_iter(self, rand, tau):
        self.propensity.choose_rxn(rand)
        self.time += tau

    @property
    def n_compartments(self):
        return len(self.compartment_lengths)

    @property
    def rxn_rates(self):
        return self.state['rates']['reaction']

    @property
    def diffusion_rates(self):
        return self.state['rates']['diffusion']

    @property
    def n_species(self):
        self.update_state_from_propensity()

        return self.state['n_species']

    @property
    def time(self):
        return self.state['time']

    @time.setter
    def time(self, value):
        self.state['time'] = value

    @property
    def propensity(self):
        if self._propensity is None:
            if self.state is not None:
                self._propensity = Propensity(self.state)

        return self._propensity

    @property
    def alpha(self):
        if self.propensity is not None:
            return self.propensity.alpha
