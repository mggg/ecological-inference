"""Test two by two ecological inference."""
import numpy as np
import scipy.stats as st
from scipy.special import logsumexp

from pyei import two_by_two


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
        two_by_two.log_binom_sum(**kwargs).eval(), log_binom_sum_in_scipy(**kwargs)
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
