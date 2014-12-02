'''
Created on Nov 29, 2014

@author: Nick Rego
'''
from __future__ import division, print_function; __metaclass__ = type

import numpy
import ssa

import logging
log = logging.getLogger('propensity')

class Propensity:
    '''Class to encapsulate system state, using numpy arrays'''

    def __init__(self, state):

        self.state = state
        self.species = None
        self.n_species = None
        self._n_compartments = None

        # Constant arrays representing differential right
        # left diffusion movements
        self.diff = None

        # Diffusion propensities for all species:
        # right propensities, then right propensities
        self.diff_prop = None

        #Cumulative sum for diff propensities
        self.diff_cum = None

        # Array of reactions
        self.rxn = None
        # Reaction propensities
        self.rxn_prop = None

        self._alpha_diff = None
        self._alpha_rxn = None

        self.init_propensities()

    def init_propensities(self):
        '''Initialize reaction propensities'''

        log.info('Loading propensity from state')

        self.species = self.state['n_species'].keys()
        log.debug('Species: {!r}'.format(self.species))

        assert (self.species == self.state['rates']['diffusion'].keys()),'Species are not ordered correctly'

        self.n_species = numpy.array(self.state['n_species'].values())
        log.debug('N Species: {!r}'.format(self.n_species))

        self.diff = numpy.array(self.state['rates']['diffusion'].values())
        log.debug('Diff array:{!r}'.format(self.diff))

        diff_prop = self.diff * self.n_species
        diff_right = diff_prop[:, :-1]
        diff_left = diff_prop[:, 1:]

        self.diff_prop = numpy.zeros((diff_prop.shape[0]*2, diff_prop.shape[1]))

        for i, specie in enumerate(self.species):
            self.diff_prop[2*i, :-1] = diff_prop[i, :-1]
            self.diff_prop[2*i+1, 1:] = diff_prop[i, 1:]

        log.debug('Diff propensity array: {!r}'.format(self.diff_prop))

        self.diff_cum = self.diff_prop.cumsum()

        self.rxn = self.rxn_prop = numpy.array(self.state['rates']['reaction'].items())

    # Choose appropriate reaction, based on rand = r*alpha (r is [0,1)])
    def choose_rxn(self, rand):
        #log.debug('r*a = {:.4f}'.format(rand))

        if rand < self.alpha_diff:
            #idx = (rand < self.diff_prop.cumsum()).argmax()
            idx = (rand < self.diff_cum).argmax()
            #log.debug('cumsum: {!r}'.format(self.diff_cum))
            #log.debug('idx: {}'.format(idx))
            self.run_diffusion(idx)

        else:
            log.debug('Choosing a reaction...')
            raise NotImplementedError

    # Run a diffusion reaction according to idx corresponding to appropriate
    # Diffusion in diff_prop.
    # Update n_species and diff_prop accordingly
    def run_diffusion(self, idx):
        row = int(idx / (self.n_compartments))
        col = idx % (self.n_compartments)

        specie_idx = int(row / 2)
        move_right = not (row % 2)

        col_from, col_to = (col, col+1) if move_right else (col, col-1)
        row_neighbor = row+1 if move_right else row-1

        #log.debug('Moving species {!r} from column {} to {}'.
        #          format(self.species[specie_idx], col_from, col_to))

        #log.debug('N Species before: {!r}'.format(self.n_species))

        self.n_species[specie_idx, col_from] -= 1
        self.n_species[specie_idx, col_to] += 1

        #log.debug('N Species after: {!r}'.format(self.n_species))

        #assert self.n_species.min() >= 0, "SHIT"

        #log.debug('diff prop before: {!r}'.format(self.diff_prop))

        # Marginal diffusion propensities
        diff_from = self.diff[specie_idx][col_from]
        diff_to = self.diff[specie_idx][col_to]

        self.diff_prop[row, col_from] -= diff_from
        self.diff_prop[row_neighbor, col_to] += diff_to

        if move_right:
            if col_to < self.n_compartments - 1:
                self.diff_prop[row, col_to] += diff_to
            if col_from > 0:
                self.diff_prop[row_neighbor, col_from] -= diff_from

        # Move left
        else:
            if col_to > 0:
                self.diff_prop[row, col_to] += diff_to
            if col_from < self.n_compartments - 1:
                self.diff_prop[row_neighbor, col_from] -= diff_from

        #log.debug('diff prop after: {!r}'.format(self.diff_prop))

        self.diff_cum = self.diff_prop.cumsum()


    @property
    def alpha_diff(self):
        return self.diff_cum[-1]


    @property
    def alpha_rxn(self):
        if self._alpha_rxn is None:
            self._alpha_rxn = self.rxn_prop.sum()

        return self._alpha_rxn

    @property
    def alpha(self):
        return self.alpha_diff + self.alpha_rxn

    @property
    def n_compartments(self):
        if self._n_compartments is None:
            self._n_compartments = self.diff.shape[1]

        return self._n_compartments
