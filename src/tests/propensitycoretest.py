'''
Created on Dec 1, 2014

@author: Nick Rego
'''
from __future__ import division, print_function; __metaclass__ = type

from system import order_state
from propensity import Propensity

import numpy
import nose


class PropensityCoreTests:

    def setUp(self):
        order_state(self.state)
        self.prop = Propensity(self.state, self.rxn_schemas)

    def tearDown(self):
        del self.prop

    def test_n_species_and_compartments_setup_correctly(self):

        prop = self.prop

        assert prop.n_species.shape[0] == len(prop.species)
        assert prop.n_species.shape[1] == prop.compartment_cnt == \
            self.expected_compartment_cnt

        assert numpy.array_equal(prop.n_species, self.expected_n_species)

    def test_alpha_sum_init_correctly(self):
        prop = self.prop

        assert prop.alpha == prop.alpha_diff + prop.alpha_rxn

    def check_state_params_unchanged(self):
        prop = self.prop

        assert prop.species == self.species
        assert prop.reactions == self.reactions
        assert prop.compartment_cnt == self.expected_compartment_cnt

        assert numpy.array_equal(prop.rxn_rates, self.rxn_rates)
        assert numpy.array_equal(prop.rxn, self.expected_rxn)
        assert numpy.array_equal(prop.rxn_stoic, self.expected_rxn_stoic)

        assert numpy.array_equal(prop.diff, self.expected_diff)


class PropensityDiffTests(PropensityCoreTests):
    '''Diffusion tests'''

    def test_diff_arrays_setup_correctly(self):

        prop = self.prop
        expected_diff_shape = (prop.specie_cnt, prop.compartment_cnt)

        assert prop.diff.shape == expected_diff_shape, \
            'Wrong diff shape: expected {!r} but got {!r}' \
            .format(expected_diff_shape, prop.diff.shape)

        expected_diff_prop_shape = (2*prop.diff.shape[0], prop.diff.shape[1])

        assert prop.diff_prop.shape == expected_diff_prop_shape, \
            'Wrong diff_prop shape: expected {!r} but got {!r}' \
            .format(expected_diff_prop_shape, prop.diff_prop.shape)

        assert numpy.array_equal(prop.diff, self.expected_diff)
        assert numpy.array_equal(prop.diff_prop, self.expected_diff_prop)

    def test_diff_cumulative_array_initialized_correctly(self):
        prop = self.prop

        assert prop.diff_cum.ndim == 1

        assert prop.diff_cum.size == (2*len(prop.species)*prop.compartment_cnt)
        assert numpy.array_equal(prop.diff_cum, self.expected_diff_cum)
        assert prop.diff_length == prop.diff_cum.size

    def test_alpha_diff_initialized_correctly(self):
        prop = self.prop

        assert prop.alpha_diff == self.expected_alpha_diff


class PropensityRxnTests(PropensityCoreTests):
    '''Reaction tests'''

    def test_rxn_arrays_setup_correctly(self):

        prop = self.prop

        expected_rxn_prop_shape = (prop.rxn_cnt, prop.compartment_cnt)

        assert prop.rxn_prop.shape == expected_rxn_prop_shape, \
            'Wrong diff_prop shape: expected {!r} but got {!r}' \
            .format(expected_rxn_prop_shape, prop.rxn_prop.shape)

        assert numpy.array_equal(prop.rxn, self.expected_rxn), \
            'Expected {!r}, got {!r}'.format(self.expected_rxn, prop.rxn)

        assert numpy.array_equal(prop.rxn_prop, self.expected_rxn_prop), \
            'Expected {!r}, got {!r}'.format(self.expected_rxn_prop, prop.rxn_prop)

    def test_rxn_cumulative_array_initialized_correctly(self):
        prop = self.prop

        assert prop.rxn_cum.ndim == 1

        assert prop.rxn_cum.size == (prop.rxn_cnt*prop.compartment_cnt)

        assert numpy.array_equal(prop.rxn_cum, self.expected_rxn_cum), \
            'Expected {!r}, got {!r}'.format(self.expected_rxn_cum, prop.rxn_cum)
        assert prop.rxn_length == prop.rxn_cum.size, \
            'Expected {!r}, got {!r}'.format(prop.rxn_cum.size, prop.rxn_length)

    def test_alpha_rxn_initialized_correctly(self):
        prop = self.prop

        assert prop.alpha_rxn == self.expected_alpha_rxn, \
            'Expected {!r}, got {!r}'.format(self.expected_alpha_rxn, prop.alpha_rxn)


# Generate an expected, updated n_species and diff_prop
def gen_diff_cases(diff_prop, diff, n_species, specie_idx):

    i = specie_idx*2
    compartments = diff_prop.shape[1]

    expected = [[None, None] for j in xrange(6)]

    # Only one compartment - no diffusion
    if compartments < 2:
        return expected

    # Edge cases - moving out of right most or out of left most compartments

    # Move left most specie to RIGHT
    expected[0] = diff_prop.copy(), n_species.copy()
    if n_species[specie_idx, 0] > 0:
        diff_prop_expected, n_species_expected = expected[0]

        n_species_expected[specie_idx, 0] -= 1
        n_species_expected[specie_idx, 1] += 1

        diff_prop_expected[i, 0] -= diff[specie_idx, 0]
        diff_prop_expected[i:i+2, 1] += diff[specie_idx, 1]

    # Move right most species to the LEFT
    expected[1] = diff_prop.copy(), n_species.copy()
    if n_species[specie_idx, -1] > 0:
        diff_prop_expected, n_species_expected = expected[1]

        n_species_expected[specie_idx, -1] -= 1
        n_species_expected[specie_idx, -2] += 1

        diff_prop_expected[i+1, -1] -= diff[specie_idx, -1]
        diff_prop_expected[i:i+2, -2] += diff[specie_idx, -2]

    if compartments < 3:
        return expected

    # Opposite edge cases (Moving into right most or into left most compartments)

    # Move species RIGHT to right most compartment
    expected[2] = diff_prop.copy(), n_species.copy()
    if n_species[specie_idx, -2] > 0:
        diff_prop_expected, n_species_expected = expected[2]

        n_species_expected[specie_idx, -2] -= 1
        n_species_expected[specie_idx, -1] += 1

        diff_prop_expected[i:i+2, -2] -= diff[specie_idx, -2]
        diff_prop_expected[i+1, -1] += diff[specie_idx, -1]

    # Move species LEFT to the left most compartment
    expected[3] = diff_prop.copy(), n_species.copy()
    if n_species[specie_idx, 1] > 0:
        diff_prop_expected, n_species_expected = expected[3]

        n_species_expected[specie_idx, 1] -= 1
        n_species_expected[specie_idx, 0] += 1

        diff_prop_expected[i:i+2, 1] -= diff[specie_idx, 1]
        diff_prop_expected[i, 0] += diff[specie_idx, 0]

    if compartments < 4:
        return expected

    # Intermediate cases - moving species right or left from intermediate bin to another intermediate bin

    # Move species RIGHT from third last bin to second last bin
    expected[4] = diff_prop.copy(), n_species.copy()
    if n_species[specie_idx, -3] > 0:
        diff_prop_expected, n_species_expected = expected[4]

        n_species_expected[specie_idx, -3] -= 1
        n_species_expected[specie_idx, -2] += 1

        diff_prop_expected[i:i+2, -3] -= diff[specie_idx, -3]
        diff_prop_expected[i:i+2, -2] += diff[specie_idx, -2]

    # Move species LEFT from third bin to second bin
    expected[5] = diff_prop.copy(), n_species.copy()
    if n_species[specie_idx, 2] > 0:
        diff_prop_expected, n_species_expected = expected[5]

        n_species_expected[specie_idx, 2] -= 1
        n_species_expected[specie_idx, 1] += 1

        diff_prop_expected[i:i+2, 2] -= diff[specie_idx, 2]
        diff_prop_expected[i:i+2, 1] += diff[specie_idx, 1]

    return expected
