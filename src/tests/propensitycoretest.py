from __future__ import division, print_function; __metaclass__ = type

from propensity import Propensity
import numpy

class PropensityCoreTests:

    def setUp(self):
        self.prop = Propensity(self.state)

    def tearDown(self):
        del self.prop

    def test_n_species_and_compartments_setup_correctly(self):

        prop = self.prop

        assert prop.n_species.shape[0] == len(prop.species)
        assert prop.n_species.shape[1] == prop.n_compartments == self.expected_n_compartments
        assert numpy.array_equal(prop.n_species, self.expected_n_species)

    def test_alpha_sum_init_correctly(self):
        prop = self.prop

        assert prop.alpha == prop.alpha_diff + prop.alpha_rxn

'''Diffusion tests'''
class PropensityDiffTests(PropensityCoreTests):

    def test_diff_arrays_setup_correctly(self):

        prop = self.prop

        assert prop.diff_prop.shape[0] == 2 * prop.diff.shape[0]
        assert prop.diff_prop.shape[1] == prop.diff.shape[1]

        assert numpy.array_equal(prop.diff, self.expected_diff)
        assert numpy.array_equal(prop.diff_prop, self.expected_diff_prop)

    def test_diff_cumulative_array_initialized_correctly(self):
        prop = self.prop

        assert prop.diff_cum.ndim == 1

        assert prop.diff_cum.size == (2*len(prop.species)*prop.n_compartments)
        assert numpy.array_equal(prop.diff_cum, self.expected_diff_cum)

    def test_alpha_diff_initialized_correctly(self):
        prop = self.prop

        assert prop.alpha_diff == self.expected_alpha_diff

# Generate an expected, updated n_species and diff_prop
# After moving species i  to the right - for corner and intermed cases
# Only works with n_compartments >= 4
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