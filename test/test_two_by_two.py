"""Test two by two ecological inference."""

import pytest
import numpy as np
import scipy.stats as st
from scipy.special import logsumexp

from pyei import two_by_two
from pyei import data
from pyei.two_by_two import TwoByTwoEI


@pytest.fixture(scope="session")
def example_two_by_two_data():
    """load santa clara data to test two by two ei and plots"""  #
    sc_data = data.Datasets.Santa_Clara.to_dataframe()
    group_fractions = np.array(sc_data["pct_e_asian_vote"])
    votes_fractions = np.array(sc_data["pct_for_hardy2"])  #
    precinct_pops = np.array(sc_data["total2"])
    demographic_group_name = "e_asian"
    candidate_name = "Hardy"
    precinct_names = sc_data["precinct"]
    return {  #
        "group_fractions": group_fractions,
        "votes_fractions": votes_fractions,
        "precint_pops": precinct_pops,  #
        "demographic_group_name": demographic_group_name,
        "candidate_name": candidate_name,
        "precinct_names": precinct_names,
    }  #


@pytest.fixture(scope="session")
def example_two_by_two_ei(example_two_by_two_data):  # pylint: disable=redefined-outer-name
    """run example two by two ei method - can use to test plotting"""
    ei_ex = TwoByTwoEI(model_name="king99_pareto_modification", pareto_scale=8, pareto_shape=2)
    ei_ex.fit(  #
        example_two_by_two_data["group_fractions"],
        example_two_by_two_data["votes_fractions"],
        example_two_by_two_data["precint_pops"],  #
        demographic_group_name=example_two_by_two_data["demographic_group_name"],
        candidate_name=example_two_by_two_data["candidate_name"],
        precinct_names=example_two_by_two_data["precinct_names"],  #
        draws=100,
        tune=100,
    )
    return ei_ex


def generate_kwargs_for_log_binom_sum():
    """Randomly generate inputs for log_binom_sum.

    Note that these should probably _not_ be random eventually, but should deterministically
    pass. For now, this will give us more confidence that the implementation matches scipy for
    a range of numeric inputs.
    """
    lower = np.random.poisson(10)
    upper = lower + np.random.poisson(100)
    obs_vote = np.random.randint(lower, upper)
    kwargs = {
        "n0_curr": np.random.randint(obs_vote, upper),
        "n1_curr": np.random.randint(obs_vote, upper),
        "b_1_curr": np.random.rand(),
        "b_2_curr": np.random.rand(),
        "prev": np.random.randn(),
    }
    kwargs["lower"] = lower
    kwargs["upper"] = upper
    kwargs["obs_vote"] = obs_vote
    return kwargs


def log_binom_sum_in_scipy(lower, upper, obs_vote, n0_curr, n1_curr, b_1_curr, b_2_curr, prev):
    """Reimplement theano logic in scipy to make sure it matches."""
    votes_withing_group_count = np.arange(lower, upper)
    return (
        logsumexp(
            st.binom(n0_curr, b_1_curr).logpmf(votes_withing_group_count)
            + st.binom(n1_curr, b_2_curr).logpmf(obs_vote - votes_withing_group_count)
        )
        + prev
    )


def test_log_binom_sum():
    kwargs = generate_kwargs_for_log_binom_sum()
    np.testing.assert_almost_equal(
        two_by_two.log_binom_sum(**kwargs).eval(), log_binom_sum_in_scipy(**kwargs), decimal=4
    )


def test_binom_conv_log_p():
    # Randomly generate a dataset of 5 precincts
    sample_data = []
    for _ in range(5):
        kwargs = generate_kwargs_for_log_binom_sum()
        kwargs.pop("prev")
        sample_data.append(kwargs)

    # Unpack and repack for binom_conv_log_p
    b_1, b_2, n_0, n_1, upper, lower, obs_votes = list(
        zip(
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
            ]
        )
    )

    theano_result = two_by_two.binom_conv_log_p(b_1, b_2, n_0, n_1, upper, lower, obs_votes).eval()

    # Now compute the result using scipy
    prev = np.array([0])
    for kwargs in sample_data:
        kwargs["prev"] = prev
        prev = log_binom_sum_in_scipy(**kwargs)

    np.testing.assert_allclose(theano_result, prev)


def test_polarization_report(example_two_by_two_ei):  # pylint: disable=redefined-outer-name
    prob_20 = example_two_by_two_ei.polarization_report(threshold=0.2)
    prob_40 = example_two_by_two_ei.polarization_report(threshold=0.4)
    thresh_95_range = example_two_by_two_ei.polarization_report(percentile=95)
    thresh_90_range = example_two_by_two_ei.polarization_report(percentile=90)

    assert prob_20 >= prob_40
    assert thresh_95_range[1] > thresh_95_range[0]
    assert thresh_95_range[1] - thresh_95_range[0] >= thresh_90_range[1] - thresh_90_range[0]
