'''
Created on Dec 1, 2014

Unit tests for propensities

@author: Nick Rego
'''
from __future__ import division, print_function; __metaclass__ = type

from propensitycoretest import gen_diff_cases, PropensityDiffTests
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

    expected_alpha_diff = numpy.sum(expected_diff_prop)

    def test_alpha_rxn_is_zero(self):
        prop = self.prop

        assert prop.alpha_rxn == 0

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

    def test_diffusion_movements(self):

        indices = (0, 7, 2, 5, 1, 6)

        for testidx, idx in enumerate(indices):
            yield self.check_move, testidx, idx
