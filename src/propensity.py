'''
Created on Nov 29, 2014

@author: Nick Rego

Encapsulate System's state, including diffusion and reaction propensities

Selects and executes proper reaction from input random
'''
from __future__ import division, print_function; __metaclass__ = type

import numpy

import logging
log = logging.getLogger('propensity')


class Propensity:
    '''Class to encapsulate system state, using numpy arrays
       Expects that all species keys from state are sorted
       (according to hash order)

       My attribute naming scheme, insofar as it exists,
             is kind of shitty - sorry about that'''

    def __init__(self, state, rxn_schemas=None, prop_mask=None, barrier=None):

        self.state = state
        self.species = None
        self.reactions = None
        self.rxn_schemas = None
        self.rxn_rates = None
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
        #   right propensities, then left propensities
        # shape: (specie_cnt, compartment_cnt)
        self.diff_prop = None

        # A mask to keep us honest
        #   Enforces even rows of diff_prop
        #   (move right) must be zero at last col,
        #   odd rows (move left) must be zero at first
        self.diff_mask = None

        # Array to represent
        #   optional 'barrier' compartments
        # shape: (specie_cnt,)
        self.barrier_mask = None
        self.barrier_idx = None

        # Cumulative sum for diff propensities
        self.diff_cum = None

        # 2 * specie_cnt * compartment_cnt
        self.diff_length = None

        # Given 'rxn_idx' occurs,
        #   Which reactions need
        #   their propensities updated?
        #   List of tuples.
        # shape: (rxn_cnt, 1)
        self.rxn_rxn_update = None

        # Given 'specie_idx' changes,
        #   Which reactions need
        #   their propensities updated?
        #   List of tuples.
        # shape: (specie_cnt, 1)
        self.specie_rxn_update = None

        # Constant array of differential reaction
        #   stoichiometries - similar to self.rxn,
        #   but for updating n_species after reaction
        #   occurs
        # shape: (rxn_cnt, specie_cnt)
        self.rxn_stoic = None

        # Stoichiometric differential reaction
        #   propensities for each species
        #   This relates to the propensity
        #   of given reaction, unlike above
        # shape: (rxn_cnt, specie_cnt)
        self.rxn = None

        # Mask for above - blocks
        #   irrelevant species
        # shape: (rxn_cnt, specie_cnt)
        self.rxn_mask = None

        # Reaction propensities, per compartment
        # shape: (rxn_count, compartment_cnt)
        self.rxn_prop = None

        self.rxn_cum = None

        self._alpha_diff = None
        self._alpha_rxn = None

        # User specified masks
        #   to optionally restrict
        #   diffusions or reactions
        #   to specific compartments
        #  numpy bool arrays, shape
        #  (specie_cnt, compartment_cnt) or
        #  (rxn_cnt, compartment_cnt), resp.
        self.prop_diff_mask = None
        self.prop_rxn_mask = None

        self.init_propensities(rxn_schemas, prop_mask)

    def init_propensities(self, rxn_schemas, prop_mask=None):
        '''Initialize reaction propensities'''

        log.info('Loading propensity from state')

        self.rxn_schemas = rxn_schemas

        self.species = sorted(self.state['n_species'].keys())
        self.reactions = sorted(self.state['rates']['reaction'].keys())
        log.debug('Species: {!r}'.format(self.species))
        log.debug('Reactions: {!r}'.format(self.reactions))

        self.n_species = numpy.array(self.state['n_species'].values(),
                                     dtype=numpy.uint64)
        log.debug('N Species: {!r}'.format(self.n_species))

        self.compartment_cnt = self.n_species.shape[1]
        assert self.compartment_cnt > 0
        self.rxn_cnt = len(self.reactions)
        self.specie_cnt = len(self.species)

        # set up user masks, if specified
        self.prop_rxn_mask = numpy.ones((self.rxn_cnt, self.compartment_cnt),
                                        dtype=numpy.bool)
        self.prop_diff_mask = numpy.ones((self.specie_cnt, self.compartment_cnt),
                                         dtype=numpy.bool)

        if prop_mask is not None:
            if 'diffusion' in prop_mask.keys():
                for i, specie in enumerate(self.species):
                    if specie in prop_mask['diffusion']:
                        self.prop_diff_mask[i] = prop_mask['diffusion'][specie]
            if 'reactions' in prop_mask.keys():
                for i, reaction in enumerate(self.reactions):
                    if reaction in prop_mask['reactions']:
                        self.prop_rxn_mask[i] = prop_mask['reactions'][reaction]

        assert (self.species == self.state['n_species'].keys() ==
                self.state['rates']['diffusion'].keys()), \
            'Species are not ordered correctly'

        assert (self.reactions ==
                self.state['rates']['reaction'].keys()), \
            'Reactions are not ordered correctly'

        self.diff = numpy.array(self.state['rates']['diffusion'].values(),
                                dtype=numpy.float64)
        # Block some species diffusion in certain compartments, if desired
        self.diff *= self.prop_diff_mask
        log.debug('Diff array:{!r}'.format(self.diff))

        barrier = self.state['barrier_spec']

        if barrier is not None:
            self.barrier_idx = barrier['barrier']
            self.barrier_mask = numpy.zeros((self.specie_cnt, 2),
                                            dtype=numpy.float64)
            for i, specie in enumerate(self.species):
                if specie in barrier:
                    barrier_diff_right = barrier[specie] / self.diff[i, self.barrier_idx]
                    barrier_diff_left = barrier[specie] / self.diff[i, self.barrier_idx+1]
                    self.barrier_mask[i, 0] = barrier_diff_right
                    self.barrier_mask[i, 1] = barrier_diff_left

        diff_prop = self.diff * self.n_species

        self.diff_prop = numpy.zeros((diff_prop.shape[0]*2, diff_prop.shape[1]),
                                     dtype=numpy.float64)
        self.diff_mask = self.diff_prop == self.diff_prop

        assert self.diff_mask.dtype == 'bool'
        assert self.diff_mask.shape == self.diff_prop.shape

        self.diff_mask[::2, -1] = False
        self.diff_mask[1::2, 0] = False

        for i, specie in enumerate(self.species):
            self.diff_prop[2*i, :-1] = diff_prop[i, :-1]
            self.diff_prop[2*i+1, 1:] = diff_prop[i, 1:]

        self.diff_prop *= self.diff_mask
        if self.barrier_idx:
            self.diff_prop[::2, self.barrier_idx] *= self.barrier_mask[:, 0]
            self.diff_prop[1::2, self.barrier_idx+1] *= self.barrier_mask[:, 1]

        log.debug('Diff propensity array: {!r}'.format(self.diff_prop))

        self.diff_cum = self.diff_prop.cumsum()
        self.diff_length = self.diff_cum.size

        # Reaction arrays
        self.rxn_rxn_update = {}
        self.specie_rxn_update = {}
        self.rxn_rates = numpy.zeros((self.rxn_cnt,),
                                     dtype=numpy.float64)
        self.rxn_stoic = numpy.zeros((self.rxn_cnt, self.specie_cnt),
                                     dtype=numpy.float64)
        self.rxn_prop = numpy.ones((self.rxn_cnt, self.compartment_cnt),
                                   dtype=numpy.float64)
        self.rxn = numpy.zeros((self.rxn_cnt, self.specie_cnt),
                               dtype=numpy.float64)

        for i, specie in enumerate(self.species):
            specie_stoic = numpy.zeros((self.specie_cnt), dtype=numpy.uint32)
            specie_stoic[i] = 1
            updates = []
            if rxn_schemas is not None:
                for j, rxn_schema in enumerate(rxn_schemas):
                    if rxn_schema.prop_change(specie_stoic, self.species):
                        updates.append(j)
            self.specie_rxn_update[i] = updates

        for i, reaction in enumerate(self.reactions):
            rxn_schema = rxn_schemas[i]
            # Hackish. Should have already been coerced to float!
            rate = float(self.state['rates']['reaction'][reaction])
            self.rxn_rates[i] = rate
            rxn_stoic = rxn_schema.get_stoichiometry(self.species)
            rxn_prop = rxn_schema.get_propensity(self.species)
            self.rxn_stoic[i] = rxn_stoic
            self.rxn[i] = rxn_prop  # * rate

            updates = []
            for j, other_reaction in enumerate(rxn_schemas):
                if other_reaction.prop_change(rxn_stoic, self.species):
                    updates.append(j)

            self.rxn_rxn_update[i] = updates

            if rxn_schema.order == 0:
                self.rxn_prop[i] = rate
            else:
                for k, specie in enumerate(self.species):
                    if rxn_prop[k] != 0:
                        self.rxn_prop[i] *= self.n_species[k] * rxn_prop[k]
                self.rxn_prop[i] *= rate

        # Block reactions from occuring in certain compartments,
        #   if desired
        self.rxn_prop *= self.prop_rxn_mask

        self.rxn_mask = self.rxn > 0

        if self.rxn_cnt == 0:
            self.rxn_prop = numpy.zeros((1, 1))
            self.rxn_cum = numpy.array([0])  # Hack if no reactions
        self.rxn_cum = self.rxn_prop.cumsum()

        self.rxn_length = self.rxn_cum.size

        log.info('Propensity succesfully initialized from state')

    # Choose appropriate reaction, based on rand = r*alpha (r is [0,1)])
    def choose_rxn(self, rand):
        #TODO: Watch out for string 'format' calls in logger - really
        # fucking slow for oft-repeated loops
        log.debug('r*a = {:.4f}'.format(rand))

        if rand <= self.alpha_diff:
            #idx = (rand < self.diff_prop.cumsum()).argmax()
            idx = (rand < self.diff_cum).argmax()
            assert idx < self.diff_length
            #log.debug('cumsum: {!r}'.format(self.diff_cum))
            #log.debug('idx: {}'.format(idx))
            self.run_diffusion(idx)

        else:
            rand -= self.alpha_diff
            idx = (rand < self.rxn_cum).argmax()
            assert idx < self.rxn_length
            #log.debug('Choosing a reaction...')
            self.run_rxn(idx)

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
            log.info("Idx: {}, specie_idx:{}, col: {}"
                     .format(idx, specie_idx, col_from))
            log.info("going to next iteration. N species: {!r}"
                     .format(self.n_species))

            return

        assert self.n_species[specie_idx, col_from] > 0
        self.n_species[specie_idx, col_from] -= 1
        self.n_species[specie_idx, col_to] += 1

        # Propensities for all reactions in col_from, col_to
        #   with specie_idx as reactant have to have propensities
        #   updated
        # Adjust rxn propensities in this
        #   compartment
        for rxn_idx_update in self.specie_rxn_update[specie_idx]:
            rate = self.rxn_rates[rxn_idx_update]
            rxn = self.rxn[rxn_idx_update]
            mask = self.rxn_mask[rxn_idx_update]
            new_prop = (self.n_species[:, col_from] * rxn)[mask]
            self.rxn_prop[rxn_idx_update, col_from] = \
                (new_prop.prod() * rate)
            self.rxn_prop[rxn_idx_update, col_from] *= \
                self.prop_rxn_mask[rxn_idx_update, col_from]
            new_prop = (self.n_species[:, col_to] * rxn)[mask]
            self.rxn_prop[rxn_idx_update, col_to] = \
                (new_prop.prod() * rate)
            self.rxn_prop[rxn_idx_update, col_to] *= \
                self.prop_rxn_mask[rxn_idx_update, col_to]

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
        self.diff_prop *= self.diff_mask
        if self.barrier_idx:
            self.diff_prop[::2, self.barrier_idx] *= self.barrier_mask[:, 0]
            self.diff_prop[1::2, self.barrier_idx+1] *= self.barrier_mask[:, 1]
        self.diff_cum = self.diff_prop.cumsum()
        self.rxn_cum = self.rxn_prop.cumsum()  # Ugh

    def run_rxn(self, idx):
        rxn_idx = int(idx / (self.compartment_cnt))  # reaction index
        compartment_idx = idx % (self.compartment_cnt)  # compartment

        # Change in n_species for compartment
        #   according to reaction stoichiometry
        stoic = self.rxn_stoic[rxn_idx]
        self.n_species[:, compartment_idx] += stoic

        # Adjust diffusion propensities for
        #   species in this compartment
        delta_diff = self.diff[:, compartment_idx] * stoic
        self.diff_prop[::2, compartment_idx] += delta_diff
        self.diff_prop[1::2, compartment_idx] += delta_diff
        self.diff_prop[:, compartment_idx] *= self.diff_mask[:, compartment_idx]
        if self.barrier_idx:
            self.diff_prop[::2, self.barrier_idx] *= self.barrier_mask[:, 0]
            self.diff_prop[1::2, self.barrier_idx+1] *= self.barrier_mask[:, 1]

        # Adjust rxn propensities in this
        #   compartment
        for rxn_idx_update in self.rxn_rxn_update[rxn_idx]:
            rate = self.rxn_rates[rxn_idx_update]
            rxn = self.rxn[rxn_idx_update]
            mask = self.rxn_mask[rxn_idx_update]
            new_prop = (self.n_species[:, compartment_idx] * rxn)[mask]
            self.rxn_prop[rxn_idx_update, compartment_idx] = \
                (new_prop.prod() * rate)
            self.rxn_prop[rxn_idx_update, compartment_idx] *= \
                self.prop_rxn_mask[rxn_idx_update, compartment_idx]

        #self.diff_prop *= self.diff_mask
        self.diff_cum = self.diff_prop.cumsum()
        self.rxn_cum = self.rxn_prop.cumsum()

    @property
    def alpha_diff(self):
        return self.diff_cum[-1]

    @property
    def alpha_rxn(self):
        return self.rxn_cum[-1]

    @property
    def alpha(self):
        return self.alpha_diff + self.alpha_rxn
