"""Test Goodmans Ecological Regression."""
import numpy as np
import pytest
from pyei.goodmans_er import *  # pylint:disable=wildcard-import,unused-wildcard-import


@pytest.fixture
def group_and_vote_fractions():
    """ Sample group and vote fractions, where every member of the demographic
        group votes for the given candidate and every non-member of the
        demographic group does not vote for the given candidate.
    """
    group_share = np.array([0, 0.2, 0.4, 0.6, 0.8])
    vote_share = np.array([0, 0.2, 0.4, 0.6, 0.8])
    return group_share, vote_share


@pytest.fixture
def group_and_vote_fractions_with_pop():
    """ Sample group and vote fractions, where every member of the demographic
        group votes for the given candidate and 10% of the demographic group's
        complement supports the given candidate (i.e., slope = 1, intercept = 0.1),
        with an exception of one precinct.
        All precincts have population 1000, except one precinct that has population
        1 and does not follow the above formula.
    """
    group_share = np.array([0, 0.2, 0.4, 0.6, 0.8, 0.9])
    vote_share = np.array([0.1, 0.3, 0.5, 0.7, 0.9, 0.9])
    populations = np.array([1000, 1000, 1000, 1000, 1000, 1])
    return group_share, vote_share, populations


def test_fit(group_and_vote_fractions):  # pylint:disable=redefined-outer-name
    model = GoodmansER()
    group_share, vote_share = group_and_vote_fractions
    model.fit(group_share, vote_share)
    np.testing.assert_almost_equal(model.intercept_, 0)
    np.testing.assert_almost_equal(model.slope_, 1)


def test_weighted_fit(group_and_vote_fractions_with_pop):  # pylint:disable=redefined-outer-name
    model = GoodmansER(is_weighted_regression=True)
    group_share, vote_share, pops = group_and_vote_fractions_with_pop
    model.fit(group_share, vote_share, pops)
    np.testing.assert_almost_equal(model.intercept_, 0.1, decimal=3)
    np.testing.assert_almost_equal(model.slope_, 1, decimal=3)


def test_summary():
    model = GoodmansER()
    model.demographic_group_name = "Trees"
    model.candidate_name = "Lorax"
    model.voting_prefs_est_ = 1.0
    model.voting_prefs_complement_est_ = 0.0
    assert (
        model.summary()
        == """Goodmans ER
        Est. fraction of Trees
        voters who voted for Lorax is
        1.000
        Est. fraction of non- Trees
        voters who voted for Lorax is
        0.000
        """
    )


def test_plot(group_and_vote_fractions):  # pylint:disable=redefined-outer-name
    model = GoodmansER()
    group_share, vote_share = group_and_vote_fractions
    model.fit(group_share, vote_share)
    _, ax = model.plot()
    x_plot, y_plot = ax.lines[0].get_xydata().T
    np.testing.assert_allclose(x_plot, y_plot, atol=1e-10)
    assert (0.0, 1.0) == ax.get_xlim()
    assert (0.0, 1.0) == ax.get_ylim()
