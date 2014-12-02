from __future__ import division, print_function; __metaclass__ = type

from propensity import Propensity
import numpy


class PropensityCoreTests:

    def setUp(self):
        self.prop = Propensity(self.state)

    def tearDown(self):
        del self.prop

    def test_n_species_array_setup_correctly(self):

        prop = self.prop

        assert len(prop.species) == prop.n_species.shape[0]
        assert prop.n_species.shape == (2, 3)
        assert numpy.array_equal(prop.n_species, self.expected_n_species)

'''Diffusion tests'''
class PropensityDiffTests(PropensityCoreTests):

    def test_n_compartments_gives_correct_result(self):
        # 3, in this case...
        assert self.prop.n_compartments == expected_n_compartments

    def test_diff_arrays_setup_correctly(self):

        prop = self.prop

        assert prop.diff_prop.shape[0] == 2 * prop.diff.shape[0]
        assert prop.diff_prop.shape[1] == prop.diff.shape[1]

        assert numpy.array_equal(prop.diff, expected_diff)
        assert numpy.array_equal(prop.diff_prop, expected_diff_prop)

    def test_diff_cumulative_array_initialized_correctly(self):
        prop = self.prop

        assert prop.diff_cum.ndim == 1

        assert prop.diff_cum.size == (2*len(prop.species)*prop.n_compartments)
        assert numpy.array_equal(prop.diff_cum, expected_diff_cum)


class TestPropensityDiffusionOnly(PropensityDiffTests):

    state = {'n_species': {'A': [10, 10, 10],
                           'B': [10, 0, 0]},
             'rates': {'diffusion': {'A': [1, 1, 1],
                                    'B': [2, 2, 2]},
                       'reaction': {}
                      }
            }

    expected_n_compartments = 3

    expected_n_species = numpy.array([[10, 10, 10],
                                      [10, 0, 0]])

    expected_diff = numpy.array([[1, 1, 1],
                                 [2, 2, 2]])

    expected_diff_prop = numpy.array([[10, 10,  0],
                                      [ 0, 10, 10],
                                      [20,  0,  0],
                                      [ 0,  0,  0]])

    expected_diff_cum = expected_diff_prop.cumsum()
