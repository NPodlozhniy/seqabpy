import numpy as np
import pytest

from seqabpy.gatsby import (
    alpha_spending_function,
    multivariate_norm_cdf,
    calculate_sequential_bounds,
    ldBounds,
    GST,
)


# ─── alpha_spending_function ─────────────────────────────────────────────────

class TestAlphaSpendingFunction:

    t = np.array([0.2, 0.4, 0.6, 0.8, 1.0])

    def test_obrien_fleming(self):
        name, spending = alpha_spending_function(self.t, iuse=1, alpha=0.05)
        assert name == "O'Brien-Fleming"
        # Very conservative early, aggressive late
        assert spending[0] < 0.001
        np.testing.assert_allclose(spending[-1], 0.05, atol=1e-6)

    def test_pocock(self):
        name, spending = alpha_spending_function(self.t, iuse=2, alpha=0.05)
        assert name == "Pocock"
        np.testing.assert_allclose(spending[-1], 0.05, atol=1e-6)
        # Pocock is more uniform than O'Brien-Fleming
        assert spending[0] > 0.01

    def test_kim_demets(self):
        name, spending = alpha_spending_function(self.t, iuse=3, phi=1, alpha=0.05)
        assert name == "Kim-DeMets"
        # With phi=1, Kim-DeMets is linear: alpha * t
        np.testing.assert_allclose(spending, 0.05 * self.t, atol=1e-10)

    def test_kim_demets_phi2(self):
        _, spending = alpha_spending_function(self.t, iuse=3, phi=2, alpha=0.05)
        # With phi=2: alpha * t^2
        np.testing.assert_allclose(spending, 0.05 * self.t**2, atol=1e-10)

    def test_hwang_shih_decani(self):
        name, spending = alpha_spending_function(self.t, iuse=4, alpha=0.05)
        assert name == "Hwang-Shih-DeCani"
        np.testing.assert_allclose(spending[-1], 0.05, atol=1e-6)

    def test_hwang_shih_decani_phi0(self):
        # phi=0 should be equivalent to linear (alpha * t)
        _, spending = alpha_spending_function(self.t, iuse=4, phi=0, alpha=0.05)
        np.testing.assert_allclose(spending, 0.05 * self.t, atol=1e-6)

    def test_haybittle_peto(self):
        name, spending = alpha_spending_function(self.t, iuse=5, alpha=0.05)
        assert name == "Haybittle-Peto"
        # Constant for interim, then alpha at final
        np.testing.assert_allclose(spending[-1], 0.05, atol=1e-6)
        # All interim values should be equal
        assert np.all(spending[:-1] == spending[0])

    def test_all_spending_monotonic(self):
        for iuse in range(1, 6):
            _, spending = alpha_spending_function(self.t, iuse=iuse, alpha=0.05)
            assert np.all(np.diff(spending) >= 0), f"iuse={iuse} is not monotonically non-decreasing"

    def test_invalid_iuse(self):
        with pytest.raises(ValueError, match="Invalid iuse value"):
            alpha_spending_function(self.t, iuse=6)


# ─── multivariate_norm_cdf ──────────────────────────────────────────────────

class TestMultivariateNormCdf:

    def test_2d_symmetric(self):
        upper = np.array([1.0, 1.0])
        lower = np.array([-1.0, -1.0])
        mean = np.array([0.0, 0.0])
        cov = np.array([[1.0, 0.5], [0.5, 1.0]])
        result = multivariate_norm_cdf(upper, lower, mean, cov, focus="accuracy")
        np.testing.assert_allclose(result, 0.4980, atol=0.001)

    def test_2d_independent(self):
        # Independent dimensions: P(-1.96<X1<1.96, -1.96<X2<1.96) ≈ 0.95^2 ≈ 0.9025
        upper = np.array([1.96, 1.96])
        lower = np.array([-1.96, -1.96])
        mean = np.array([0.0, 0.0])
        cov = np.array([[1.0, 0.0], [0.0, 1.0]])
        result = multivariate_norm_cdf(upper, lower, mean, cov, focus="accuracy")
        np.testing.assert_allclose(result, 0.95**2, atol=0.001)

    def test_performance_mode_fallback(self):
        # "performance" mode falls back to "accuracy" on SciPy >= 1.16
        upper = np.array([1.0, 1.0])
        lower = np.array([-1.0, -1.0])
        mean = np.array([0.0, 0.0])
        cov = np.array([[1.0, 0.5], [0.5, 1.0]])
        result = multivariate_norm_cdf(upper, lower, mean, cov, focus="performance")
        np.testing.assert_allclose(result, 0.4980, atol=0.001)

    def test_invalid_focus(self):
        with pytest.raises(ValueError, match="Invalid `focus` value"):
            multivariate_norm_cdf(
                np.array([1.0]), np.array([-1.0]),
                np.array([0.0]), np.array([[1.0]]),
                focus="invalid"
            )


# ─── calculate_sequential_bounds ─────────────────────────────────────────────

class TestCalculateSequentialBounds:

    def test_upper_bounds_only(self):
        """Without beta, only upper bounds (alpha spending) are returned."""
        incremental_alpha, upper_bounds = calculate_sequential_bounds(
            np.linspace(1 / 5, 1, 5), alpha=0.05
        )
        assert len(upper_bounds) == 5
        assert len(incremental_alpha) == 5
        # Upper bounds should be decreasing (O'Brien-Fleming default)
        assert np.all(np.diff(upper_bounds) < 0)
        # Incremental alpha should sum to ~alpha
        np.testing.assert_allclose(np.sum(incremental_alpha), 0.05, atol=1e-4)

    def test_with_beta_non_binding(self):
        expected_lower = np.array([-3.021, -1.415, -0.596, -0.057, 0.351, 0.682, 0.964, 1.212, 1.434, 1.795])
        expected_upper = np.array([6.088, 4.229, 3.396, 2.906, 2.579, 2.342, 2.16, 2.015, 1.895, 1.795])
        lower, upper = calculate_sequential_bounds(
            np.linspace(1 / 10, 1, 10), alpha=0.05, beta=0.2
        )
        np.testing.assert_allclose(lower, expected_lower, atol=0.001)
        np.testing.assert_allclose(upper, expected_upper, atol=0.001)

    def test_with_beta_binding(self):
        lower, upper = calculate_sequential_bounds(
            np.linspace(1 / 5, 1, 5), alpha=0.05, beta=0.2, binding=True
        )
        expected_lower = np.array([-1.472, -0.084, 0.651, 1.166, 1.63])
        expected_upper = np.array([4.229, 2.888, 2.298, 1.954, 1.63])
        np.testing.assert_allclose(lower, expected_lower, atol=0.001)
        np.testing.assert_allclose(upper, expected_upper, atol=0.001)
        # Last lower and upper should be equal
        np.testing.assert_allclose(lower[-1], upper[-1], atol=0.001)

    def test_custom_beta_spending(self):
        lower, upper = calculate_sequential_bounds(
            np.linspace(1 / 5, 1, 5), alpha=0.05, beta=0.2, beta_iuse=2, beta_phi=1
        )
        expected_lower = np.array([-0.267, 0.382, 0.88, 1.295, 1.74])
        expected_upper = np.array([4.229, 2.888, 2.298, 1.962, 1.74])
        np.testing.assert_allclose(lower, expected_lower, atol=0.001)
        np.testing.assert_allclose(upper, expected_upper, atol=0.001)

    def test_last_bounds_equal_with_beta(self):
        """Last lower bound should equal last upper bound when beta is specified."""
        lower, upper = calculate_sequential_bounds(
            np.linspace(1 / 5, 1, 5), alpha=0.05, beta=0.2
        )
        np.testing.assert_allclose(lower[-1], upper[-1], atol=0.001)


# ─── ldBounds ────────────────────────────────────────────────────────────────

class TestLdBounds:

    @pytest.mark.parametrize("alpha,expected_bounds", [
        (0.025, [3.929, 2.67, 1.981]),
        (0.05, [3.393, 2.281, 1.68]),
        (0.01, [4.559, 3.127, 2.337]),
    ])
    def test_obrien_fleming(self, alpha, expected_bounds):
        result = ldBounds(t=np.array([0.3, 0.6, 1.0]), alpha=alpha)
        np.testing.assert_allclose(
            result["upper.bounds"], expected_bounds, atol=0.001
        )
        np.testing.assert_allclose(result["overall.alpha"], alpha, atol=1e-4)

    def test_pocock(self):
        result = ldBounds(t=np.array([0.3, 0.6, 1.0]), alpha=0.025, iuse=2)
        np.testing.assert_allclose(
            result["upper.bounds"], [2.312, 2.321, 2.269], atol=0.001
        )

    def test_kim_demets(self):
        result = ldBounds(t=np.array([0.3, 0.6, 1.0]), alpha=0.025, iuse=3)
        np.testing.assert_allclose(
            result["upper.bounds"], [2.432, 2.336, 2.177], atol=0.001
        )

    def test_result_keys(self):
        result = ldBounds(t=np.array([0.5, 1.0]), alpha=0.05)
        assert set(result.keys()) == {
            "time.points", "alpha.spending", "overall.alpha",
            "upper.bounds", "nominal.alpha"
        }

    def test_invalid_alpha(self):
        with pytest.raises(ValueError, match="Alpha must be between 0 and 1"):
            ldBounds(t=np.array([0.5, 1.0]), alpha=1.5)

    def test_invalid_iuse(self):
        with pytest.raises(ValueError, match="Invalid iuse value"):
            ldBounds(t=np.array([0.5, 1.0]), iuse=6)

    def test_invalid_phi_obrien_fleming(self):
        with pytest.raises(ValueError, match="Phi must be at least 1"):
            ldBounds(t=np.array([0.5, 1.0]), iuse=1, phi=0.5)

    def test_invalid_phi_kim_demets(self):
        with pytest.raises(ValueError, match="Phi must be positive"):
            ldBounds(t=np.array([0.5, 1.0]), iuse=3, phi=0)

    def test_invalid_phi_hsd(self):
        with pytest.raises(ValueError, match="Phi must be non-negative"):
            ldBounds(t=np.array([0.5, 1.0]), iuse=4, phi=-1)

    def test_invalid_phi_haybittle_peto(self):
        with pytest.raises(ValueError, match="Haybittle-Peto"):
            ldBounds(t=np.array([0.5, 1.0]), iuse=5, phi=0.5)


# ─── GST ─────────────────────────────────────────────────────────────────────

class TestGST:

    def test_exact_match(self):
        """When actual == expected, GST returns same as ldBounds."""
        expected = np.array([0.3, 0.6, 1.0])
        bounds = GST(actual=expected, expected=expected, alpha=0.025)
        ld_result = ldBounds(t=expected, alpha=0.025)
        np.testing.assert_allclose(bounds, ld_result["upper.bounds"], atol=1e-4)

    def test_oversampling(self):
        bounds = GST(
            actual=np.array([0.3, 0.6, 1.2]),
            expected=np.array([0.3, 0.6, 1]),
            alpha=0.025,
        )
        np.testing.assert_allclose(bounds, [3.929, 2.67, 1.989], atol=0.001)

    def test_undersampling(self):
        bounds = GST(
            actual=np.array([0.3, 0.6, 0.8]),
            expected=np.array([0.3, 0.6, 1]),
            alpha=0.025,
        )
        np.testing.assert_allclose(bounds, [3.929, 2.67, 1.969], atol=0.001)

    def test_integer_inputs(self):
        """Integer inputs should generate uniform peeking strategies."""
        bounds = GST(actual=3, expected=3, alpha=0.025)
        ld_result = ldBounds(t=np.array([1/3, 2/3, 1.0]), alpha=0.025)
        np.testing.assert_allclose(bounds, ld_result["upper.bounds"], atol=0.001)

    def test_mismatched_types(self):
        with pytest.raises(ValueError, match="same type"):
            GST(actual=3, expected=np.array([0.3, 0.6, 1.0]), alpha=0.025)

    def test_wrong_beginning(self):
        with pytest.raises(ValueError, match="same beginning"):
            GST(
                actual=np.array([0.2, 0.6, 1.0]),
                expected=np.array([0.3, 0.6, 1.0]),
                alpha=0.025,
            )

    def test_invalid_expected_fractions(self):
        with pytest.raises(ValueError, match="between 0 and 1"):
            GST(
                actual=np.array([0.3, 0.6, 1.5]),
                expected=np.array([0.3, 0.6, 1.5]),
                alpha=0.025,
            )
