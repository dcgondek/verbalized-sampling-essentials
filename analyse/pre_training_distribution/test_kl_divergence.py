import unittest
import math
from calc_kl_divergence import calc_kl_divergence, calc_kl_divergence_uniform


class TestCalcKLDivergence(unittest.TestCase):
    """Test cases for calc_kl_divergence basic functionality"""

    def test_identical_distributions(self):
        """KL divergence should be 0 for identical distributions"""
        gt_data = {"Alaska": 0.5, "Rhode Island": 0.3, "California": 0.2}
        data = {"Alaska": 0.5, "Rhode Island": 0.3, "California": 0.2}
        result = calc_kl_divergence(gt_data, data)
        self.assertAlmostEqual(result, 0.0, places=6)

    def test_different_distributions(self):
        """KL divergence should be positive for different distributions"""
        gt_data = {"Alaska": 0.5, "Rhode Island": 0.3, "California": 0.2}
        data = {"Alaska": 0.2, "Rhode Island": 0.3, "California": 0.5}
        result = calc_kl_divergence(gt_data, data)
        self.assertGreater(result, 0)

    def test_no_overlap(self):
        """KL divergence should be 0 when there's no overlap between keys"""
        gt_data = {"Alaska": 0.5, "Rhode Island": 0.5}
        data = {"California": 0.7, "Texas": 0.3}
        result = calc_kl_divergence(gt_data, data)
        self.assertEqual(result, 0.0)

    def test_partial_overlap(self):
        """KL divergence should only consider overlapping keys"""
        gt_data = {"Alaska": 0.4, "Rhode Island": 0.3, "California": 0.3}
        data = {"Alaska": 0.5, "Rhode Island": 0.5, "Texas": 0.0}
        result = calc_kl_divergence(gt_data, data)
        self.assertGreater(result, 0)


    def test_unnormalized_inputs(self):
        """Function should handle unnormalized probability distributions"""
        gt_data = {"Alaska": 0.5, "Rhode Island": 0.5}  # Sum = 1.0
        data = {"Alaska": 0.6, "Rhode Island": 0.6}  # Sum = 1.2
        result = calc_kl_divergence(gt_data, data)
        self.assertIsInstance(result, float)
        self.assertEqual(result, 0)

    def test_single_element(self):
        """Test with single element distributions"""
        gt_data = {"Alaska": 1.0}
        data = {"Alaska": 1.0}
        result = calc_kl_divergence(gt_data, data)
        self.assertAlmostEqual(result, 0.0, places=6)



class TestCalcKLDivergenceUniform(unittest.TestCase):
    """Test cases for calc_kl_divergence_uniform basic functionality"""

    def test_uniform_distribution(self):
        """KL divergence should be 0 for uniform distribution"""
        data = {"Alaska": 0.25, "Rhode Island": 0.25, "California": 0.25, "Texas": 0.25}
        result = calc_kl_divergence_uniform(data)
        self.assertAlmostEqual(result, 0.0, places=6)

    def test_non_uniform_distribution(self):
        """KL divergence should be positive for non-uniform distributions"""
        data = {"Alaska": 0.7, "Rhode Island": 0.1, "California": 0.1, "Texas": 0.1}
        result = calc_kl_divergence_uniform(data)
        self.assertGreater(result, 0)




if __name__ == '__main__':
    unittest.main()
