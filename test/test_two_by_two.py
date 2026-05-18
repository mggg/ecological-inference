"""Test two by two ecological inference."""

import numpy as np
import pytest
import scipy.stats as st
from scipy.special import logsumexp

from pyei import two_by_two

# Fixed log_binom_sum cases, generated once with np.random.seed(0).
# Hardcoded here so the comparison against scipy is inspectable
# and independent of numpy's RNG algorithm across versions.
LOG_BINOM_SUM_CASES = [
    {
        "n0_curr": 102,
        "n1_curr": 40,
        "b_1_curr": 0.3843817072926998,
        "b_2_curr": 0.2975346065444723,
        "prev": -0.055035123059621,
        "lower": 10,
        "upper": 113,
        "obs_vote": 19,
    },
    {
        "n0_curr": 107,
        "n1_curr": 44,
        "b_1_curr": 0.6481718720511972,
        "b_2_curr": 0.36824153984054797,
        "prev": -0.10731045224678325,
        "lower": 10,
        "upper": 109,
        "obs_vote": 35,
    },
    {
        "n0_curr": 117,
        "n1_curr": 95,
        "b_1_curr": 0.7805291762864555,
        "b_2_curr": 0.11827442586893322,
        "prev": -0.9607546117337881,
        "lower": 17,
        "upper": 130,
        "obs_vote": 66,
    },
    {
        "n0_curr": 110,
        "n1_curr": 94,
        "b_1_curr": 0.13521817340545206,
        "b_2_curr": 0.3241410077932141,
        "prev": 0.37692697499528294,
        "lower": 17,
        "upper": 114,
        "obs_vote": 92,
    },
    {
        "n0_curr": 79,
        "n1_curr": 49,
        "b_1_curr": 0.6976311959272649,
        "b_2_curr": 0.06022547162926983,
        "prev": 1.2302906807277207,
        "lower": 6,
        "upper": 103,
        "obs_vote": 48,
    },
]


def log_binom_sum_in_scipy(
    lower, upper, obs_vote, n0_curr, n1_curr, b_1_curr, b_2_curr, prev
):
    """Reimplement the theano logic in scipy as an independent reference."""
    votes_within_group_count = np.arange(lower, upper)
    return (
        logsumexp(
            st.binom(n0_curr, b_1_curr).logpmf(votes_within_group_count)
            + st.binom(n1_curr, b_2_curr).logpmf(obs_vote - votes_within_group_count)
        )
        + prev
    )


@pytest.mark.parametrize("kwargs", LOG_BINOM_SUM_CASES)
def test_log_binom_sum(kwargs):
    np.testing.assert_almost_equal(
        two_by_two._log_binom_sum(**kwargs).eval(),
        log_binom_sum_in_scipy(**kwargs),
        decimal=4,
    )


def test_binom_conv_log_p():
    sample_data = [
        {k: v for k, v in case.items() if k != "prev"} for case in LOG_BINOM_SUM_CASES
    ]

    b_1, b_2, n_0, n_1, upper, lower, obs_votes = zip(
        *[
            (
                v["b_1_curr"],
                v["b_2_curr"],
                v["n0_curr"],
                v["n1_curr"],
                v["upper"],
                v["lower"],
                v["obs_vote"],
            )
            for v in sample_data
        ],
        strict=True,
    )

    theano_result = two_by_two._binom_conv_log_p(
        b_1, b_2, n_0, n_1, upper, lower, obs_votes
    ).eval()

    prev = np.array([0])
    for kwargs in sample_data:
        kwargs["prev"] = prev
        prev = log_binom_sum_in_scipy(**kwargs)

    np.testing.assert_allclose(theano_result, prev)


@pytest.mark.slow
def test_precinct_level_estimates_shapes(example_two_by_two_ei):
    """Returns (num_precincts x 2 x 2, num_precincts x 2 x 2 x 2)."""
    means, intervals = example_two_by_two_ei.precinct_level_estimates()
    num_precincts = len(example_two_by_two_ei.precinct_pops)
    assert means.shape == (num_precincts, 2, 2)
    assert intervals.shape == (num_precincts, 2, 2, 2)


@pytest.mark.slow
def test_precinct_level_estimates_candidate_pair_sums_to_one(example_two_by_two_ei):
    """For each (precinct, group), the candidate / complement means sum to 1.

    2x2 has exactly two candidate columns: vote for the candidate, and the
    complement (1 - p). They must always sum to 1.
    """
    means, _ = example_two_by_two_ei.precinct_level_estimates()
    np.testing.assert_allclose(means.sum(axis=2), 1.0, atol=1e-9)


@pytest.mark.slow
def test_precinct_level_estimates_intervals_bracket_means(example_two_by_two_ei):
    """Posterior mean must lie within the 95% credible interval for every cell."""
    means, intervals = example_two_by_two_ei.precinct_level_estimates()
    lower = intervals[..., 0]
    upper = intervals[..., 1]
    assert np.all(lower <= means + 1e-9)
    assert np.all(means - 1e-9 <= upper)
    assert np.all(lower >= 0.0)
    assert np.all(upper <= 1.0)


@pytest.mark.slow
def test_precinct_level_estimates_complement_intervals_mirror(example_two_by_two_ei):
    """Complement (column 1) interval is the reflection of the column-0
    interval around 0.5, with endpoints preserved as (lower, upper):
    lower_complement = 1 - upper_original, upper_complement = 1 - lower_original.
    """
    _, intervals = example_two_by_two_ei.precinct_level_estimates()
    for group_idx in (0, 1):
        np.testing.assert_allclose(
            intervals[:, group_idx, 1, 0], 1 - intervals[:, group_idx, 0, 1]
        )
        np.testing.assert_allclose(
            intervals[:, group_idx, 1, 1], 1 - intervals[:, group_idx, 0, 0]
        )


@pytest.mark.slow
def test_two_by_two_fit_recovers_known_truth():
    """End-to-end posterior recovery on synthetic ground truth.

    Generates precinct-level vote counts from known per-group preferences
    (b_1 = 0.75 for the demographic group, b_2 = 0.20 for the complement),
    then fits ``TwoByTwoEI`` and verifies the posterior mean of
    ``sampled_voting_prefs`` recovers those truths within tolerance.

    This guards the wiring between the model definition, the sampler, and
    the posterior-aggregation step in ``calculate_sampled_voting_prefs`` —
    none of which is covered by the per-primitive scipy tests.
    """
    rng = np.random.default_rng(0)
    num_precincts = 40
    precinct_pops = np.full(num_precincts, 600, dtype=np.int64)

    # Varied group fractions so both slopes are identifiable.
    group_fraction = rng.uniform(0.15, 0.85, size=num_precincts)
    b_1_true = 0.75
    b_2_true = 0.20

    # Per-precinct true vote rate, then binomial draw for the observed votes.
    p_per_precinct = b_1_true * group_fraction + b_2_true * (1 - group_fraction)
    vote_counts = rng.binomial(precinct_pops, p_per_precinct)
    votes_fraction = vote_counts / precinct_pops

    ei = two_by_two.TwoByTwoEI(
        model_name="king99_pareto_modification", pareto_scale=8, pareto_shape=2
    )
    ei.fit(
        group_fraction,
        votes_fraction,
        precinct_pops,
        demographic_group_name="synth_group",
        candidate_name="synth_cand",
        draws=400,
        tune=400,
        random_seed=0,
    )

    posterior_b1 = ei.sampled_voting_prefs[0].mean()
    posterior_b2 = ei.sampled_voting_prefs[1].mean()
    # Tolerance budget: short chain (400 draws) plus the multimodality of
    # ER-style likelihoods means recovery within 0.05 is realistic without
    # being so tight that minor sampler-version drift flakes the test.
    np.testing.assert_allclose(posterior_b1, b_1_true, atol=0.05)
    np.testing.assert_allclose(posterior_b2, b_2_true, atol=0.05)


@pytest.mark.slow
def test_polarization_report(example_two_by_two_ei):
    prob_20 = example_two_by_two_ei.polarization_report(threshold=0.2)
    prob_40 = example_two_by_two_ei.polarization_report(threshold=0.4)
    thresh_95_range = example_two_by_two_ei.polarization_report(percentile=95)
    thresh_90_range = example_two_by_two_ei.polarization_report(percentile=90)

    assert prob_20 >= prob_40
    assert thresh_95_range[1] > thresh_95_range[0]
    assert (
        thresh_95_range[1] - thresh_95_range[0]
        >= thresh_90_range[1] - thresh_90_range[0]
    )
