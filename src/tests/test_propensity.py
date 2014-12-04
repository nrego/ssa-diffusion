'''
Created on Dec 1, 2014

Unit tests for propensities

@author: Nick Rego
'''
from __future__ import division, print_function; __metaclass__ = type

from propensitycoretest import gen_diff_cases, PropensityDiffTests, PropensityRxnTests

from system import ReactionSchema

import numpy
import nose


class TestPropensityDiffusionOnly(PropensityDiffTests):

    state = {'n_species': {'A': [10, 10, 10, 10],
                           'B': [10, 0, 0, 0]},
             'rates': {'diffusion': {'A': [1, 1, 1, 1],
                                     'B': [2, 2, 2, 2]},
                       'reaction': {}
                       }
             }

    species = sorted(state['n_species'].keys())
    reactions = sorted(state['rates']['reaction'].keys())

    rxn_rates = [state['rates']['reaction'][rxn]
                 for rxn in reactions]
    rxn_rates = numpy.array(rxn_rates)
    rxn_schemas = None

    expected_compartment_cnt = 4

    expected_n_species = numpy.array([state['n_species']['A'],
                                      state['n_species']['B']])

    expected_diff = numpy.array([[1, 1, 1, 1],
                                 [2, 2, 2, 2]])

    expected_diff_prop = numpy.array([[10, 10, 10,  0],
                                      [ 0, 10, 10, 10],
                                      [20,  0, 0,  0],
                                      [ 0,  0, 0, 0]])


    expected_diff_cum = expected_diff_prop.cumsum()

    diff_tests_A = gen_diff_cases(expected_diff_prop, expected_diff, expected_n_species, 0)
    diff_tests_B = gen_diff_cases(expected_diff_prop, expected_diff, expected_n_species, 1)

    expected_rxn = numpy.zeros((0, 2))
    expected_rxn_stoic = numpy.zeros((0, 2))

    expected_alpha_diff = numpy.sum(expected_diff_prop)

    def test_alpha_rxn_is_zero(self):
        prop = self.prop

        assert prop.alpha_rxn == 0

    # Run diffusion according to certain index, idx,
    #   check against expected diffusion
    def check_move(self, testidx, idx):
        prop = self.prop

        alpha_rxn_before = prop.alpha_rxn

        testcase = self.diff_tests_A[testidx]
        prop.run_diffusion(idx)

        expected_alpha_diff = numpy.sum(testcase[0])

        assert numpy.array_equal(prop.diff_prop, testcase[0]), "Expected:\n {!r}, \ngot:\n {!r}".format(testcase[0], prop.diff_prop)
        assert numpy.array_equal(prop.n_species, testcase[1]), "Expected:\n {!r}, \ngot:\n {!r}".format(testcase[1], prop.n_species)

        assert numpy.array_equal(prop.diff_cum, testcase[0].cumsum())

        assert prop.alpha_diff == expected_alpha_diff, 'Alpha_diff wrong after diffusion!'
        assert prop.alpha_rxn == alpha_rxn_before, 'Alpha reaction changed after diffusion'

        assert prop.alpha == alpha_rxn_before + expected_alpha_diff, 'Alpha not updated correctly after diffusion'

        self.check_state_params_unchanged()

    def test_diffusion_movements(self):

        indices = (0, 7, 2, 5, 1, 6)

        for testidx, idx in enumerate(indices):
            yield self.check_move, testidx, idx


#@nose.SkipTest
class TestPropensityReactionOnly(PropensityRxnTests):

    # Rxns:
    # A + B -> A
    # 0 -> B
    state = {'n_species': {'A': [10],
                           'B': [5],
                           'C': [2]},
             'rates': {'diffusion': {'A': [0],
                                     'B': [0],
                                     'C': [0]},
                       'reaction': {'deg': 0.1,
                                    'prod': 1}
                       }
             }

    species = sorted(state['n_species'].keys())
    reactions = sorted(state['rates']['reaction'].keys())

    rxn_rates = [state['rates']['reaction'][rxn]
                 for rxn in reactions]
    rxn_rates = numpy.array(rxn_rates)

    deg = {'reactants': ['A', 'C'],
           'products': ['A']}
    prod = {'reactants': [],
            'products': ['C']}

    rxn_schemas = [ReactionSchema('deg', deg),
                   ReactionSchema('prod', prod)]

    expected_compartment_cnt = 1

    expected_n_species = numpy.array([state['n_species']['A'],
                                      state['n_species']['B'],
                                      state['n_species']['C']])

    # (rxn_cnt, specie_cnt) -
    # reaction differential propensities
    # will automatically multiply by reaction rates,
    #  below
    expected_rxn = numpy.array([[1, 0, 1],
                                [0, 0, 0]], dtype=numpy.float32)

    expected_rxn_stoic = numpy.array([[0, 0, -1],
                                      [0, 0,  1]])

    # (rxn_cnt, compartment_cnt)
    # Initial reaction propensities,
    # per compartment
    expected_rxn_prop = numpy.array([[2],
                                     [1]])

    expected_rxn_cum = numpy.array([2, 3])

    expected_alpha_rxn = 3

    expected_diff = numpy.zeros((3,1))
    expected_alpha_diff = 0

    def test_alpha_diff_is_zero(self):
        prop = self.prop

        assert prop.alpha_diff == 0

    def test_deg_rxn(self):
        prop = self.prop

        idx = 0
        expected_n_species = numpy.array([[10],
                                          [5],
                                          [1]])
        expected_rxn_prop = numpy.array([[1],
                                         [1]])
        expected_rxn_cum = numpy.array([1, 2])
        prop.run_rxn(idx)

        numpy.array_equal(prop.n_species, expected_n_species)
        numpy.array_equal(prop.rxn_prop, expected_rxn_prop)
        numpy.array_equal(prop.rxn_cum, expected_rxn_cum)

        assert prop.alpha == prop.alpha_rxn == 2

        self.check_state_params_unchanged()

    def test_prod_rxn(self):
        prop = self.prop

        idx = 1
        expected_n_species = numpy.array([[10],
                                          [5],
                                          [3]])
        expected_rxn_prop = numpy.array([[3],
                                         [1]])
        expected_rxn_cum = numpy.array([3, 4])
        prop.run_rxn(idx)

        numpy.array_equal(prop.n_species, expected_n_species)
        numpy.array_equal(prop.rxn_prop, expected_rxn_prop)
        numpy.array_equal(prop.rxn_cum, expected_rxn_cum)

        assert prop.alpha == prop.alpha_rxn == 4

        self.check_state_params_unchanged()
