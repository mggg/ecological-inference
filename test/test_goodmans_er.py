"""Test Goodmans Ecological Regression."""

# pylint:disable=redefined-outer-name
import numpy as np
import pytest

from pyei import data
from pyei.goodmans_er import GoodmansER, GoodmansERBayes


@pytest.fixture(scope="session")
def example_two_by_two_data():  # noqa: ANN201
    """Load santa clara data to test two by two ei and plots"""
    sc_data = data.Datasets.Santa_Clara.to_dataframe()
    group_fractions = np.array(sc_data["pct_e_asian_vote"])
    votes_fractions = np.array(sc_data["pct_for_hardy2"])
    precinct_pops = np.array(sc_data["total2"])
    demographic_group_name = "e_asian"
    candidate_name = "Hardy"
    precinct_names = sc_data["precinct"]
    return {
        "group_fractions": group_fractions,
        "votes_fractions": votes_fractions,
        "precint_pops": precinct_pops,
        "demographic_group_name": demographic_group_name,
        "candidate_name": candidate_name,
        "precinct_names": precinct_names,
    }


@pytest.fixture
def group_and_vote_fractions():  # noqa: ANN201
    """Sample group and vote fractions, where every member of the demographic

    group votes for the given candidate and every non-member of the
    demographic group does not vote for the given candidate.
    """
    group_share = np.array([0, 0.2, 0.4, 0.6, 0.8])
    vote_share = np.array([0, 0.2, 0.4, 0.6, 0.8])
    return group_share, vote_share


@pytest.fixture
def group_and_vote_fractions_with_pop():  # noqa: ANN201
    """Sample group and vote fractions, where every member of the demographic

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


@pytest.fixture
def goodmans_er_bayes_examples(example_two_by_two_data):  # noqa: ANN001,ANN201
    """Run Bayesian Goodman's ER"""
    ex = example_two_by_two_data
    bayes_goodman_ei_weighted = GoodmansERBayes(
        "goodman_er_bayes", weighted_by_pop=True, sigma=1
    )
    bayes_goodman_ei_weighted.fit(
        ex["group_fractions"],
        ex["votes_fractions"],
        ex["precint_pops"],
        demographic_group_name=ex["demographic_group_name"],
        candidate_name=ex["candidate_name"],
        tune=2000,
    )

    bayes_goodman_ei_unweighted = GoodmansERBayes(
        "goodman_er_bayes", weighted_by_pop=False, sigma=1
    )
    bayes_goodman_ei_unweighted.fit(
        ex["group_fractions"],
        ex["votes_fractions"],
        ex["precint_pops"],
        demographic_group_name=ex["demographic_group_name"],
        candidate_name=ex["candidate_name"],
    )
    return {
        "bayes_goodman_ei_weighted": bayes_goodman_ei_weighted,
        "bayes_goodman_ei_unweighted": bayes_goodman_ei_unweighted,
    }


def test_fit(group_and_vote_fractions):  # noqa: ANN001,ANN201,D103
    model = GoodmansER()
    group_share, vote_share = group_and_vote_fractions
    model.fit(group_share, vote_share)
    np.testing.assert_almost_equal(model.intercept_, 0)
    np.testing.assert_almost_equal(model.slope_, 1)


def test_weighted_fit(group_and_vote_fractions_with_pop):  # noqa: ANN001,ANN201,D103
    model = GoodmansER(is_weighted_regression=True)
    group_share, vote_share, pops = group_and_vote_fractions_with_pop
    model.fit(group_share, vote_share, pops)
    np.testing.assert_almost_equal(model.intercept_, 0.1, decimal=3)
    np.testing.assert_almost_equal(model.slope_, 1, decimal=3)


def test_summary():  # noqa: ANN201,D103
    model = GoodmansER()
    model.demographic_group_name = "Trees"
    model.candidate_name = "Lorax"
    model.voting_prefs_est_ = 1.0
    model.voting_prefs_complement_est_ = 0.0
    expected_summary = """Goodmans ER
        Est. fraction of Trees
        voters who voted for Lorax is
        1.000
        Est. fraction of non- Trees
        voters who voted for Lorax is
        0.000
        """
    assert model.summary() == expected_summary
    model.is_weighted_regression = True
    expected_summary = """Goodmans ER, weighted by population
        Est. fraction of Trees
        voters who voted for Lorax is
        1.000
        Est. fraction of non- Trees
        voters who voted for Lorax is
        0.000
        """
    assert model.summary() == expected_summary


def test_plot(group_and_vote_fractions):  # noqa: ANN001,ANN201,D103
    model = GoodmansER()
    group_share, vote_share = group_and_vote_fractions
    model.fit(group_share, vote_share)
    _, ax = model.plot()
    x_plot, y_plot = ax.lines[0].get_xydata().T
    np.testing.assert_allclose(x_plot, y_plot, atol=1e-10)
    assert (0.0, 1.0) == ax.get_xlim()
    assert (0.0, 1.0) == ax.get_ylim()


def test_goodman_er_bayes_posterior_means(goodmans_er_bayes_examples):  # noqa: ANN001,ANN201,D103
    goodmans_er_bayes_weighted = goodmans_er_bayes_examples["bayes_goodman_ei_weighted"]
    np.testing.assert_almost_equal(
        goodmans_er_bayes_weighted.sampled_voting_prefs[0].mean(), 0.840, decimal=2
    )
    np.testing.assert_almost_equal(
        goodmans_er_bayes_weighted.sampled_voting_prefs[1].mean(), 0.240, decimal=2
    )

    goodmans_er_bayes_unweighted = goodmans_er_bayes_examples[
        "bayes_goodman_ei_unweighted"
    ]
    np.testing.assert_almost_equal(
        goodmans_er_bayes_unweighted.sampled_voting_prefs[0].mean(), 0.835, decimal=2
    )
    np.testing.assert_almost_equal(
        goodmans_er_bayes_unweighted.sampled_voting_prefs[1].mean(), 0.244, decimal=2
    )


def test_goodman_er_bayes_bounds(goodmans_er_bayes_examples):  # noqa: ANN001,ANN201,D103
    goodmans_er_bayes_example = goodmans_er_bayes_examples["bayes_goodman_ei_weighted"]
    (
        _,
        _,
        lower_bounds,
        upper_bounds,
    ) = goodmans_er_bayes_example.compute_credible_int_for_line()

    assert all(upper_bounds) <= 1.0
    assert all(upper_bounds) >= 0

    assert all(lower_bounds) <= 1.0
    assert all(lower_bounds) >= 0


def test_goodman_er_bayes_plot(goodmans_er_bayes_examples):  # noqa: ANN001,ANN201,D103
    ax = goodmans_er_bayes_examples["bayes_goodman_ei_weighted"].plot()
    assert ax is not None
