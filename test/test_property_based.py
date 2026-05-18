"""Property-based tests for numerical primitives.

Hypothesis explores the input space more thoroughly than hand-written
examples can. We keep ``max_examples`` small so these tests stay in the
fast lane; the goal is to catch obvious property violations, not to
fuzz to convergence.
"""

import numpy as np
import scipy.stats as st
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as hst

from pyei.distribution_utils import non_central_hypergeometric_sample
from pyei.greiner_quinn_gibbs_sampling import (
    _get_initial_internal_count_sample,
    _omega_to_theta,
    _theta_to_omega,
)

# Reusable per-module config: small example count keeps these in the fast lane,
# and we suppress function-scoped-fixture warnings because none of these tests
# use fixtures.
PROPERTY_SETTINGS = settings(
    max_examples=40,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)


# ---------------------------------------------------------------------------
# _theta_to_omega / _omega_to_theta
# ---------------------------------------------------------------------------


@PROPERTY_SETTINGS
@given(
    num_precincts=hst.integers(min_value=1, max_value=6),
    r=hst.integers(min_value=1, max_value=4),
    c=hst.integers(min_value=2, max_value=5),
)
def test_theta_omega_roundtrip(num_precincts, r, c):
    """omega → theta → omega must be the identity (modulo float epsilon).

    theta_to_omega(theta) = log(theta[..., :-1] / theta[..., -1]) and
    omega_to_theta is its inverse for any positive theta with rows summing
    to 1.
    """
    theta = st.dirichlet.rvs(np.ones(c), size=(num_precincts, r))
    omega = _theta_to_omega(theta)
    omega_flat = omega.reshape(num_precincts, r * (c - 1))
    # _omega_to_theta has the documented side effect of reshaping its input;
    # pass a copy so we don't perturb omega_flat for any later use.
    theta_reconstructed = _omega_to_theta(omega_flat.copy(), r, c)
    np.testing.assert_allclose(theta_reconstructed, theta, atol=1e-9)


@PROPERTY_SETTINGS
@given(
    num_precincts=hst.integers(min_value=1, max_value=6),
    r=hst.integers(min_value=1, max_value=4),
    c=hst.integers(min_value=2, max_value=5),
)
def test_theta_to_omega_shape_drops_last_candidate(num_precincts, r, c):
    """omega has c-1 entries per (precinct, group); the last candidate is
    the reference category and is implicit."""
    theta = st.dirichlet.rvs(np.ones(c), size=(num_precincts, r))
    omega = _theta_to_omega(theta)
    assert omega.shape == (num_precincts, r, c - 1)


# ---------------------------------------------------------------------------
# _get_initial_internal_count_sample
# ---------------------------------------------------------------------------


@PROPERTY_SETTINGS
@given(
    num_precincts=hst.integers(min_value=1, max_value=4),
    r=hst.integers(min_value=2, max_value=4),
    c=hst.integers(min_value=2, max_value=4),
    seed=hst.integers(min_value=0, max_value=2**31 - 1),
)
def test_initial_internal_count_preserves_both_marginals(num_precincts, r, c, seed):
    """Initialized internal counts must satisfy both the group-sum and
    candidate-sum marginal constraints — this is the precondition the
    Gibbs sampler relies on for correctness."""
    rng = np.random.default_rng(seed)
    precinct_pops = rng.integers(10, 100, size=num_precincts)

    group_fractions = rng.dirichlet(np.ones(r), size=num_precincts)
    vote_fractions = rng.dirichlet(np.ones(c), size=num_precincts)

    # Construct integer counts that exactly sum to precinct_pops on each axis.
    # We do this by rounding and then adjusting one entry to absorb the remainder
    # — straightforward and avoids interaction with the autouse stdlib RNG seed.
    group_counts = np.round(group_fractions * precinct_pops[:, None]).astype(np.int64)
    group_counts[:, -1] += precinct_pops - group_counts.sum(axis=1)
    vote_counts = np.round(vote_fractions * precinct_pops[:, None]).astype(np.int64)
    vote_counts[:, -1] += precinct_pops - vote_counts.sum(axis=1)

    # If adjustment pushed any cell negative, skip — that would not be a
    # well-formed input that the sampler is required to handle.
    if (group_counts < 0).any() or (vote_counts < 0).any():
        return

    samp = _get_initial_internal_count_sample.py_func(
        group_counts, vote_counts, precinct_pops
    )
    # Group marginal: sum across candidates == group counts.
    np.testing.assert_array_equal(samp.sum(axis=2), group_counts)
    # Candidate marginal: sum across groups == vote counts.
    np.testing.assert_array_equal(samp.sum(axis=1), vote_counts)


# ---------------------------------------------------------------------------
# non_central_hypergeometric_sample
# ---------------------------------------------------------------------------


@PROPERTY_SETTINGS
@given(
    n1=hst.integers(min_value=1, max_value=50),
    n2=hst.integers(min_value=1, max_value=50),
    psi=hst.floats(
        min_value=0.05, max_value=20.0, allow_nan=False, allow_infinity=False
    ),
)
def test_nchg_sample_in_support(n1, n2, psi):
    """The sample must lie in [max(0, m1 - n2), min(n1, m1)] for any m1
    achievable by Binom(n1+n2, p) — we cover the interior range here."""
    # Pick m1 mid-range so both bounds are non-trivial.
    m1 = max(1, min(n1, n2))
    samp = non_central_hypergeometric_sample.py_func(n1, n2, m1, psi)
    assert max(0, m1 - n2) <= samp <= min(n1, m1)
