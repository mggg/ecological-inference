"""test r-by-c specific plotting"""
import pytest
import numpy as np
from pyei import data
from pyei.plot_utils import plot_precinct_scatterplot
from pyei.r_by_c import RowByColumnEI


@pytest.fixture(scope="session")
def example_r_by_c_data():
    """trimmed santa clara dataset"""
    sc_data = data.Datasets.Santa_Clara.to_dataframe()
    sc_data = sc_data.iloc[:10, :]
    precinct_pops = np.array(sc_data["total2"])
    votes_fractions = np.array(sc_data[["pct_for_hardy2", "pct_for_kolstad2", "pct_for_nadeem2"]]).T
    candidate_names = ["Hardy", "Kolstad", "Nadeem"]
    group_fractions = np.array(
        sc_data[["pct_ind_vote", "pct_e_asian_vote", "pct_non_asian_vote"]]
    ).T
    demographic_group_names = ["ind", "e_asian", "non_asian"]
    return {
        "group_fractions": group_fractions,
        "votes_fractions": votes_fractions,
        "precinct_pops": precinct_pops,
        "demographic_group_names": demographic_group_names,
        "candidate_names": candidate_names,
    }


def example_r_by_c_ei(example_r_by_c_data):  # pylint: disable=redefined-outer-name
    """Run this to generate an EI instance"""

    ei_ex = RowByColumnEI(model_name="multinomial-dirichlet")
    ei_ex.fit(
        example_r_by_c_data["group_fractions"],
        example_r_by_c_data["votes_fractions"],
        example_r_by_c_data["precinct_pops"],
        example_r_by_c_data["demographic_group_names"],
        example_r_by_c_data["candidate_names"],
        draws=100,
        tune=100,
    )
    return ei_ex


@pytest.fixture(scope="session")
def two_r_by_c_ei_runs(example_r_by_c_data):  # pylint: disable=redefined-outer-name
    """use the EI Factory to fix two EI instances for our tests"""
    example_ei_r_by_c_1 = example_r_by_c_ei(example_r_by_c_data)
    example_ei_r_by_c_2 = example_r_by_c_ei(example_r_by_c_data)
    return [example_ei_r_by_c_1, example_ei_r_by_c_2]


def test_ei_r_by_c_precinct_scatterplot(two_r_by_c_ei_runs):  # pylint: disable=redefined-outer-name
    all_demographics_ax = plot_precinct_scatterplot(
        two_r_by_c_ei_runs, ["Run 1", "Run 2"], "Kolstad"
    )
    just_ind_ax = plot_precinct_scatterplot(two_r_by_c_ei_runs, ["Run 1", "Run 2"], "Nadeem", "ind")
    assert all_demographics_ax is not None
    assert just_ind_ax is not None


def test_ei_r_by_c_boxplots(two_r_by_c_ei_runs):  # pylint: disable=redefined-outer-name
    # TODO: maybe uncouple this to test the plot utils piece alone
    example_r_by_c_ei = two_r_by_c_ei_runs[0]  # pylint: disable=redefined-outer-name
    assert example_r_by_c_ei.plot_boxplots() is not None
    assert example_r_by_c_ei.plot_boxplots(plot_by="group") is not None
    with pytest.raises(ValueError):
        example_r_by_c_ei.plot_boxplots(plot_by="grupo")


def test_ei_r_by_c_kdes(two_r_by_c_ei_runs):  # pylint: disable=redefined-outer-name
    # TODO: maybe uncouple this to test the plot utils piece alone
    example_r_by_c_ei = two_r_by_c_ei_runs[0]  # pylint: disable=redefined-outer-name
    assert example_r_by_c_ei.plot_kdes(plot_by="candidate") is not None
    assert example_r_by_c_ei.plot_kdes(plot_by="group") is not None
    with pytest.raises(ValueError):
        example_r_by_c_ei.plot_kdes(plot_by="grupo")


def test_ei_r_by_c_intervals_by_precinct(
    two_r_by_c_ei_runs,
):  # pylint: disable=redefined-outer-name
    example_r_by_c_ei = two_r_by_c_ei_runs[0]
    assert example_r_by_c_ei.plot_intervals_by_precinct("e_asian", "Kolstad") is not None
    with pytest.raises(ValueError):
        example_r_by_c_ei.plot_intervals_by_precinct("e_asian", "Kolstaad")
        example_r_by_c_ei.plot_intervals_by_precinct("ibnd", "Hardy")
