from pyei import data
import pytest
import numpy as np
from pyei.r_by_c import RowByColumnEI


@pytest.fixture(scope="session")
def example_r_by_c_data():
    """"""
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


@pytest.fixture(scope="session")
def example_r_by_c_ei(example_r_by_c_data):
    """"""
    ei = RowByColumnEI(model_name="multinomial-dirichlet")
    ei.fit(
        example_r_by_c_data["group_fractions"],
        example_r_by_c_data["votes_fractions"],
        example_r_by_c_data["precinct_pops"],
        example_r_by_c_data["demographic_group_names"],
        example_r_by_c_data["candidate_names"],
    )
    return ei


def test_ei_r_by_c_boxplots(example_r_by_c_ei):
    # TODO: maybe uncouple this to test the plot utils piece alone
    example_r_by_c_ei.plot_boxplots()
    example_r_by_c_ei.plot_boxplots(plot_by="group")


def test_ei_r_by_c_kdes(example_r_by_c_ei):
    # TODO: maybe uncouple this to test the plot utils piece alone
    example_r_by_c_ei.plot_kdes(plot_by="candidate")
    example_r_by_c_ei.plot_kdes(plot_by="group")


def test_ei_r_by_c_intervals_by_precinct(example_r_by_c_ei):
    example_r_by_c_ei.plot_intervals_by_precinct("e_asian", "Kolstad")
