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
