"""Tests for pyei.distribution_utils.

Non-central hypergeometric distribution (Liao & Rosen 2001): samples
``y1 | y1 + y2 = m1`` where ``y1 ~ Binom(n1, p1)``, ``y2 ~ Binom(n2, p2)``
and ``psi = p1 (1-p2) / p2 (1-p1)`` is the odds ratio.

When psi = 1, the distribution collapses to the standard hypergeometric.
"""

import numpy as np
import pytest
import scipy.stats as st

from pyei.distribution_utils import non_central_hypergeometric_sample


def _sample_n(n1, n2, m1, psi, n, py_func=False):
    """Draw n samples; pick numba-jitted or pure-Python implementation."""
    fn = (
        non_central_hypergeometric_sample.py_func
        if py_func
        else non_central_hypergeometric_sample
    )
    return np.array([fn(n1, n2, m1, psi) for _ in range(n)])


@pytest.mark.parametrize("py_func", [False, True], ids=["jit", "py_func"])
def test_support_bounds(py_func):
    """Samples must lie in [max(0, m1-n2), min(n1, m1)]."""
    n1, n2, m1, psi = 10, 5, 7, 1.0
    samples = _sample_n(n1, n2, m1, psi, 200, py_func=py_func)
    lower = max(0, m1 - n2)
    upper = min(n1, m1)
    assert samples.min() >= lower
    assert samples.max() <= upper


@pytest.mark.parametrize("py_func", [False, True], ids=["jit", "py_func"])
@pytest.mark.parametrize(
    "n1, n2, m1",
    [
        (10, 5, 7),
        (20, 30, 15),
        (50, 50, 40),
        (100, 5, 7),
    ],
)
def test_psi_1_matches_central_hypergeometric(n1, n2, m1, py_func):
    """psi=1 ⇒ classical hypergeometric. Compare means against scipy."""
    n_samples = 5000
    samples = _sample_n(n1, n2, m1, 1.0, n_samples, py_func=py_func)

    # scipy.stats.hypergeom signature: hypergeom(M, n, N) where
    # M = population (n1+n2), n = successes in pop (n1), N = draws (m1).
    analytic_mean = st.hypergeom.mean(n1 + n2, n1, m1)
    analytic_var = st.hypergeom.var(n1 + n2, n1, m1)
    expected_se = np.sqrt(analytic_var / n_samples)

    # 5-sigma envelope: false-positive rate ~6e-7, comfortably below
    # the flake budget for a 4-case x 2-impl parametrize.
    np.testing.assert_allclose(samples.mean(), analytic_mean, atol=5 * expected_se)


@pytest.mark.parametrize("py_func", [False, True], ids=["jit", "py_func"])
def test_psi_greater_than_one_shifts_mass_up(py_func):
    """psi > 1 should push the mean above the hypergeometric mean."""
    n1, n2, m1 = 20, 20, 15
    n_samples = 3000
    samples_eq = _sample_n(n1, n2, m1, 1.0, n_samples, py_func=py_func).mean()
    samples_hi = _sample_n(n1, n2, m1, 5.0, n_samples, py_func=py_func).mean()
    assert samples_hi > samples_eq


@pytest.mark.parametrize("py_func", [False, True], ids=["jit", "py_func"])
def test_psi_less_than_one_shifts_mass_down(py_func):
    """psi < 1 should push the mean below the hypergeometric mean."""
    n1, n2, m1 = 20, 20, 15
    n_samples = 3000
    samples_eq = _sample_n(n1, n2, m1, 1.0, n_samples, py_func=py_func).mean()
    samples_lo = _sample_n(n1, n2, m1, 0.2, n_samples, py_func=py_func).mean()
    assert samples_lo < samples_eq


def test_jit_and_py_func_distributions_agree():
    """The numba-jitted and pure-Python paths sample from the same
    distribution; numba uses its own RNG state, so we compare empirical
    moments rather than per-sample equality."""
    n_samples = 5000
    samples_jit = _sample_n(10, 10, 7, 2.0, n_samples, py_func=False)
    samples_py = _sample_n(10, 10, 7, 2.0, n_samples, py_func=True)
    # 5-sigma difference-of-means test under the null that they share a mean.
    pooled_se = np.sqrt(
        samples_jit.var(ddof=1) / n_samples + samples_py.var(ddof=1) / n_samples
    )
    assert abs(samples_jit.mean() - samples_py.mean()) < 5 * pooled_se


def test_mode_at_lower_bound_branch():
    """Exercise the mode == ll fast path: m1 small forces mode to lower bound."""
    samples = _sample_n(10, 10, 1, 1.0, 200)
    assert samples.min() == 0
    assert samples.max() <= 1


def test_mode_at_upper_bound_branch():
    """Exercise the mode == uu fast path: heavy psi with m1 == n1 pins
    most mass to the upper bound n1."""
    samples = _sample_n(5, 10, 5, 100.0, 500)
    # With psi=100 and m1==n1, the upper-bound mode dominates but doesn't
    # saturate — empirically ~65–75%. Threshold at 0.5 to test the direction
    # without making the test fragile to the exact mass distribution.
    assert (samples == 5).mean() > 0.5
