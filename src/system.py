'''
Created on Nov 26, 2014

@author: Nick Rego
'''
from __future__ import division, print_function; __metaclass__ = type

import numpy
from collections import OrderedDict

import ssa

from propensity import Propensity

import logging
log = logging.getLogger('system')


class ReactionSchema:

    def __init__(self, rxn_name, rxn_data):
        self.name = rxn_name

        try:
            self.reactants = [reactant for reactant in sorted(rxn_data['reactants'])]
        except KeyError:
            self.reactants = []
        try:
            self.products = [product for product in sorted(rxn_data['products'])]
        except KeyError:
            self.products = []

    def __repr__(self):
        return 'Reaction: {}, reactants: {!r}, products: {!r}'.format(self.name, self.reactants, self.products)

    # Check that all species involved in this reaction are in list 'species'
    def check_reaction(self, species):

        for reactant in self.reactants:
            if reactant not in species:
                raise ValueError("reactant {!r} not found".format(reactant))
        for product in self.products:
            if product not in species:
                raise ValueError("product {!r} not found".format(product))

        return True

    # Return (species_cnt,) array of stoichiometric changes after reaction,
    # indexed according to 'species'
    def get_stoichiometry(self, species):

        stoic_arr = numpy.zeros((len(species)))

        for i, specie in enumerate(species):
            if specie in self.reactants:
                stoic_arr[i] -= 1
            if specie in self.products:
                stoic_arr[i] += 1

        return stoic_arr


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
        reactions = config.get(['system', 'spec', 'reactions'], {})

        self.reactions = []
        self.reaction_names = []

        for rxn_name in sorted(reactions.keys()):
            rxn_data = config.require(['system', 'spec', 'reactions', rxn_name])

            rxn_schema = ReactionSchema(rxn_name, rxn_data)
            rxn_schema.check_reaction(self.species)
            self.reactions.append(rxn_schema)
            self.reaction_names.append(rxn_schema.name)

        assert self.reaction_names == sorted(self.reaction_names)


    def __init__(self, rc=None):
        self.rc = rc or ssa.rc

        # system spec stuff
        self.compartment_bounds = None
        self.compartment_lengths = None
        self.species = None
        self.reactions = None
        self.reaction_names = None
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
        reaction_rates = {reaction: 0.0 for reaction in self.reaction_names}
        diffusion_rates = {specie: [0.0 for i in xrange(self.n_compartments)] for specie in self.species}
        rates = {
            'reaction': OrderedDict(sorted(reaction_rates.items(), key=lambda t: t[0])),
            'diffusion': OrderedDict(sorted(diffusion_rates.items(), key=lambda t: t[0]))
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

        species_arr = {specie: [0 for i in self.compartment_lengths]
                                    for specie in self.species}

        # Initialize 'state' - i.e. what changes as simulation runs
        self.state = {'time': 0.0,
                      'n_species': OrderedDict(sorted(species_arr.items(), key=lambda t: t[0])),
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

        self.rc.pstatus('Compartments (mm):')
        self.rc.pstatus('  N compartments: {!r}'.format(self.n_compartments))
        self.rc.pstatus('  Compartment bounds: {!r}'.format(self.compartment_bounds))
        self.rc.pstatus('  Compartment lengths: {!r}'.format(self.compartment_lengths))

        self.rc.pstatus('Rates (1/s):')
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
                self._propensity = Propensity(self.state, self.reactions)

        return self._propensity

    @property
    def alpha(self):
        if self.propensity is not None:
            return self.propensity.alpha
