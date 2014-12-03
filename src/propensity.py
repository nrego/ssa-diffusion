'''
Created on Nov 29, 2014

@author: Nick Rego

Encapsulate System's state, including diffusion and reaction propensities

Selects and executes proper reaction from input random
'''
from __future__ import division, print_function; __metaclass__ = type

import numpy
import ssa

import logging
log = logging.getLogger('propensity')

class Propensity:
    '''Class to encapsulate system state, using numpy arrays
       Expects that all species keys from state are sorted (according to hash order)'''

    def __init__(self, state, rxn_schemas=None):

        self.state = state
        self.species = None
        self.reactions = None
        # N species array
        # shape: (specie_cnt,)
        self.n_species = None

        # Constant attributes
        self.compartment_cnt = None
        self.specie_cnt = None
        self.rxn_cnt = None

        # Constant arrays representing differential right
        #   left diffusion movements (i.e. d)
        # shape: (specie_cnt, compartment_cnt)
        self.diff = None

        # Diffusion propensities for all species:
        #   right propensities, then right propensities
        # shape: (specie_cnt, compartment_cnt)
        self.diff_prop = None

        #Cumulative sum for diff propensities
        self.diff_cum = None
        self.diff_length = None

        # Constant array of differential reaction
        #   propensities - depends on stoichiometry
        #   of relevant reaction and reaction rate
        # shape: (rxn_cnt, specie_cnt)
        self.rxn = None
        # Reaction propensities, per compartment
        # shape: (rxn_count, compartment_cnt)
        self.rxn_prop = None
        self.rxn_cum = None

        self._alpha_diff = None
        self._alpha_rxn = None

        self.init_propensities(rxn_schemas)

    def init_propensities(self, rxn_schemas):
        '''Initialize reaction propensities'''

        log.info('Loading propensity from state')

        self.species = sorted(self.state['n_species'].keys())
        self.reactions = sorted(self.state['rates']['reaction'].keys())
        log.debug('Species: {!r}'.format(self.species))

        self.n_species = numpy.array(self.state['n_species'].values(),
                                     dtype=numpy.uint32)
        log.debug('N Species: {!r}'.format(self.n_species))

        self.compartment_cnt = self.n_species.shape[1]
        self.rxn_cnt = len(self.reactions)
        self.specie_cnt = len(self.species)

        assert (self.species == self.state['n_species'].keys() ==
                self.state['rates']['diffusion'].keys()),'Species are not ordered correctly'
        assert (self.reactions ==
                self.state['rates']['reaction'].keys()), 'Reactions are not ordered correctly'

        self.diff = numpy.array(self.state['rates']['diffusion'].values(),
                                dtype=numpy.float32)
        log.debug('Diff array:{!r}'.format(self.diff))

        diff_prop = self.diff * self.n_species

        self.diff_prop = numpy.zeros((diff_prop.shape[0]*2, diff_prop.shape[1]),
                                     dtype=numpy.float32)

        for i, specie in enumerate(self.species):
            self.diff_prop[2*i, :-1] = diff_prop[i, :-1]
            self.diff_prop[2*i+1, 1:] = diff_prop[i, 1:]

        log.debug('Diff propensity array: {!r}'.format(self.diff_prop))

        self.diff_cum = self.diff_prop.cumsum()

        # Reaction arrays
        self.rxn = numpy.zeros((self.rxn_cnt, self.specie_cnt),
                               dtype=numpy.float32)

        for i, reaction in enumerate(self.reactions):
            rxn_schema = rxn_schemas[i]
            rate = self.state['rates']['reaction'][reaction]
            rxn_stoic = rxn_schema.get_stoichiometry(self.species)

            self.rxn[i] = rxn_stoic * rate

        self.rxn_prop = self.rxn.dot(self.n_species)

        self.diff_length = len(self.diff_cum)

    # Choose appropriate reaction, based on rand = r*alpha (r is [0,1)])
    def choose_rxn(self, rand):
        #log.debug('r*a = {:.4f}'.format(rand))

        if rand <= self.alpha_diff:
            #idx = (rand < self.diff_prop.cumsum()).argmax()
            idx = (rand < self.diff_cum).argmax()
            assert idx < self.diff_length
            #log.debug('cumsum: {!r}'.format(self.diff_cum))
            #log.debug('idx: {}'.format(idx))
            self.run_diffusion(idx)

        else:
            rand -= self.alpha_diff
            #log.debug('Choosing a reaction...')
            raise NotImplementedError

    # Run a diffusion reaction according to idx corresponding to appropriate
    # Diffusion in diff_prop.
    # Update n_species and diff_prop accordingly
    def run_diffusion(self, idx):
        row = int(idx / (self.compartment_cnt))
        col = idx % (self.compartment_cnt)

        specie_idx = int(row / 2)
        move_right = not (row % 2)

        col_from, col_to = (col, col+1) if move_right else (col, col-1)
        row_neighbor = row+1 if move_right else row-1

        # Illegal species movement, but handle gracefully for sake of unit tests
        if self.n_species[specie_idx, col_from] == 0:
            log.info("Trying to move a species where one doesn't exist:")
            log.info("Idx: {}, specie_idx:{}, col: {}".format(idx, specie_idx, col_from))
            log.info("going to next iteration. N species: {!r}".format(self.n_species))

            return

        self.n_species[specie_idx, col_from] -= 1
        self.n_species[specie_idx, col_to] += 1

        # Marginal diffusion propensities
        diff_from = self.diff[specie_idx][col_from]
        diff_to = self.diff[specie_idx][col_to]

        self.diff_prop[row, col_from] -= diff_from
        self.diff_prop[row_neighbor, col_to] += diff_to

        if move_right:
            if col_to < self.compartment_cnt - 1:
                self.diff_prop[row, col_to] += diff_to
            if col_from > 0:
                self.diff_prop[row_neighbor, col_from] -= diff_from

        # Move left
        else:
            if col_to > 0:
                self.diff_prop[row, col_to] += diff_to
            if col_from < self.compartment_cnt - 1:
                self.diff_prop[row_neighbor, col_from] -= diff_from

        #log.debug('diff prop after: {!r}'.format(self.diff_prop))

        self.diff_cum = self.diff_prop.cumsum()

    @property
    def alpha_diff(self):
        return self.diff_cum[-1] or 0

    @property
    def alpha_rxn(self):
        if self._alpha_rxn is None:
            self._alpha_rxn = numpy.sum(self.rxn_prop)

        return self._alpha_rxn

    @property
    def alpha(self):
        return self.alpha_diff + self.alpha_rxn
