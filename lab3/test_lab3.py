import unittest

import numpy as np

import analysis
from params import Params
from discrete_params import (
    DEPENDENT,
    INDEPENDENT,
    MATRICES,
    VALUES_X,
    VALUES_Y,
    characteristics,
    marginals,
)
from sampling import generate_pairs, generate_pairs_discrete


class ParamsTests(unittest.TestCase):
    def test_cdf_and_inverse_agree(self):
        u = np.linspace(0.01, 0.99, 50)
        np.testing.assert_allclose(Params.cdf_x(Params.cdf_x_inv(u)), u)
        np.testing.assert_allclose(Params.cdf_y(Params.cdf_y_inv(u)), u)

    def test_density_is_zero_outside_the_domain(self):
        self.assertEqual(Params.f(np.array(-1.0), np.array(1.0)), 0.0)
        self.assertEqual(Params.f_x(np.array(-0.5)), 0.0)
        self.assertEqual(Params.cdf_x(np.array(-0.5)), 0.0)

    def test_joint_density_is_product_of_conditional_and_marginal(self):
        x = np.array([0.3, 1.2])
        y = np.array([0.7, 2.0])
        np.testing.assert_allclose(
            Params.f(x, y), Params.f_y_given_x(y, x) * Params.f_x(x)
        )


class SamplingTests(unittest.TestCase):
    def test_sample_moments_are_close_to_theory(self):
        x, y = generate_pairs(100_000, np.random.default_rng(1))
        self.assertAlmostEqual(x.mean(), Params.MEAN_X, delta=0.02)
        self.assertAlmostEqual(y.var(ddof=1), Params.VAR_Y, delta=0.05)
        self.assertAlmostEqual(np.corrcoef(x, y)[0, 1], Params.RHO, delta=0.02)


class AnalysisTests(unittest.TestCase):
    def test_estimates_of_a_small_sample(self):
        est = analysis.estimates(np.array([1.0, 2.0, 3.0, 4.0]))
        self.assertAlmostEqual(est.mean, 2.5)
        self.assertAlmostEqual(est.variance, 5.0 / 3.0)
        self.assertLess(est.mean_interval[0], est.mean)
        self.assertGreater(est.variance_interval[1], est.variance)
        self.assertIsNone(analysis.estimates(np.array([1.0])))

    def test_variance_interval_covers_the_true_value_for_exponential_data(self):
        rng = np.random.default_rng(4)
        trials = 300
        covered = 0
        for _ in range(trials):
            est = analysis.estimates(rng.exponential(1.0, 2000))
            covered += est.variance_interval[0] <= 1.0 <= est.variance_interval[1]
        self.assertGreater(covered / trials, 0.90)
        self.assertLess(covered / trials, 0.99)

    def test_tests_do_not_reject_true_hypotheses(self):
        sample = np.random.default_rng(5).exponential(1.0, 10_000)
        self.assertFalse(analysis.mean_test(sample, 1.0).rejected)
        self.assertFalse(analysis.variance_test(sample, 1.0).rejected)

    def test_tests_reject_false_hypotheses(self):
        sample = np.random.default_rng(6).exponential(1.0, 10_000)
        self.assertTrue(analysis.mean_test(sample, 1.2).rejected)
        self.assertTrue(analysis.variance_test(sample, 1.5).rejected)
        self.assertIsNone(analysis.mean_test(np.array([1.0]), 1.0))
        self.assertIsNone(analysis.variance_test(np.array([1.0]), 1.0))

    def test_correlation_of_a_linear_relation_is_high(self):
        x = np.arange(10.0)
        corr = analysis.correlation(x, 2.0 * x + 1.0 + np.sin(x))
        self.assertGreater(corr.r, 0.95)
        self.assertLess(corr.p_value, 1e-3)
        self.assertLess(corr.interval[0], corr.r)

    def test_independent_sample_passes_the_independence_test(self):
        x, y = generate_pairs(10_000, np.random.default_rng(2))
        edges = Params.cdf_x_inv(np.arange(1, 20) / 20)
        test = analysis.independence_test(x, y, edges, edges)
        self.assertEqual(test.dof, 399)
        self.assertAlmostEqual(test.min_expected, 25.0)
        self.assertFalse(test.rejected)

    def test_dependent_sample_fails_the_independence_test(self):
        x, _ = generate_pairs(10_000, np.random.default_rng(3))
        edges = Params.cdf_x_inv(np.arange(1, 20) / 20)
        self.assertTrue(analysis.independence_test(x, x, edges, edges).rejected)

    def test_density_histogram_integrates_to_one(self):
        heights, edges = analysis.density_histogram(np.array([0.1, 0.2, 0.9]), 0.0, 1.0, 5)
        self.assertAlmostEqual(float(np.sum(heights * np.diff(edges))), 1.0)

    def test_density_histogram_2d_volume_is_the_share_of_points_inside(self):
        x = np.array([0.1, 0.6, 0.9, 5.0])
        y = np.array([0.2, 0.7, 0.4, 5.0])
        heights, x_edges, y_edges = analysis.density_histogram_2d(x, y, (0.0, 1.0), (0.0, 1.0), 4)
        volume = heights.sum() * (x_edges[1] - x_edges[0]) * (y_edges[1] - y_edges[0])
        self.assertAlmostEqual(float(volume), 0.75)

    def test_class_indices_cover_the_whole_axis(self):
        idx = analysis.class_indices(np.array([-5.0, 0.5, 1.0, 9.0]), np.array([1.0, 2.0]))
        self.assertEqual(idx.tolist(), [0, 0, 1, 2])


class DiscreteTests(unittest.TestCase):
    def test_matrices_are_distributions(self):
        for matrix in MATRICES.values():
            self.assertAlmostEqual(float(matrix.sum()), 1.0)
            self.assertTrue((matrix > 0.0005).all())

    def test_independent_matrix_factorizes_and_shares_marginals(self):
        p_x, p_y = marginals(INDEPENDENT)
        np.testing.assert_allclose(INDEPENDENT, np.outer(p_x, p_y))
        np.testing.assert_allclose(marginals(INDEPENDENT)[0], marginals(DEPENDENT)[0])
        np.testing.assert_allclose(marginals(INDEPENDENT)[1], marginals(DEPENDENT)[1])

    def test_theoretical_characteristics(self):
        mean_x, mean_y, var_x, var_y, rho = characteristics(DEPENDENT)
        self.assertAlmostEqual(mean_x, 3.0)
        self.assertAlmostEqual(mean_y, 3.0)
        self.assertAlmostEqual(var_x, 1.83)
        self.assertAlmostEqual(var_y, 1.83)
        self.assertAlmostEqual(rho, 1.11 / 1.83)
        self.assertAlmostEqual(characteristics(INDEPENDENT)[4], 0.0)

    def test_generated_frequencies_match_the_matrix(self):
        x, y = generate_pairs_discrete(200_000, DEPENDENT, VALUES_X, VALUES_Y, np.random.default_rng(7))
        observed = analysis.contingency_table(x, y, VALUES_X, VALUES_Y)
        np.testing.assert_allclose(observed / observed.sum(), DEPENDENT, atol=0.004)

    def test_independence_test_tells_the_matrices_apart(self):
        rng = np.random.default_rng(8)
        results = {}
        for name, matrix in MATRICES.items():
            x, y = generate_pairs_discrete(10_000, matrix, VALUES_X, VALUES_Y, rng)
            results[name] = analysis.independence_test_table(
                analysis.contingency_table(x, y, VALUES_X, VALUES_Y)
            )
        self.assertEqual(results["Dependent"].dof, 16)
        self.assertTrue(results["Dependent"].rejected)
        self.assertFalse(results["Independent"].rejected)

    def test_independence_test_drops_empty_rows_and_needs_a_real_table(self):
        observed = np.array([[5, 7, 0], [0, 0, 0], [6, 4, 0]])
        self.assertEqual(analysis.independence_test_table(observed).dof, 1)
        self.assertIsNone(analysis.independence_test_table(np.array([[3, 4], [0, 0]])))


if __name__ == "__main__":
    unittest.main()
