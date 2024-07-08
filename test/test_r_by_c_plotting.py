"""test r-by-c specific plotting"""

import pytest
import numpy as np
from pyei import data
from pyei.plot_utils import plot_precinct_scatterplot
from pyei.r_by_c import RowByColumnEI

# from pyei.io_utils import to_netcdf, from_netcdf


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


@pytest.fixture(scope="session")
def example_r_by_c_data_asym():
    """trimmed santa clara dataset with r not equal to c"""
    sc_data = data.Datasets.Santa_Clara.to_dataframe()
    sc_data = sc_data.iloc[:10, :]
    precinct_pops = np.array(sc_data["total2"])
    votes_fractions = np.array(sc_data[["pct_for_hardy2", "pct_for_kolstad2", "pct_for_nadeem2"]]).T
    candidate_names = ["Hardy", "Kolstad", "Nadeem"]
    group_fractions = np.array(sc_data[["pct_asian_vote", "pct_non_asian_vote"]]).T
    demographic_group_names = ["asian", "non_asian"]
    return {
        "group_fractions": group_fractions,
        "votes_fractions": votes_fractions,
        "precinct_pops": precinct_pops,
        "demographic_group_names": demographic_group_names,
        "candidate_names": candidate_names,
    }


def example_r_by_c_ei(example_r_by_c_data, model_name):  # pylint: disable=redefined-outer-name
    """Run this to generate an EI instance"""

    ei_ex = RowByColumnEI(model_name=model_name)
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
def two_r_by_c_ei_runs(
    example_r_by_c_data, example_r_by_c_data_asym
):  # pylint: disable=redefined-outer-name
    """use the EI Factory to fix two EI instances for our tests"""
    example_ei_r_by_c_1 = example_r_by_c_ei(example_r_by_c_data, "multinomial-dirichlet")
    example_ei_r_by_c_2 = example_r_by_c_ei(example_r_by_c_data, "multinomial-dirichlet-modified")
    example_ei_r_by_c_asym = example_r_by_c_ei(example_r_by_c_data_asym, "multinomial-dirichlet")
    return [example_ei_r_by_c_1, example_ei_r_by_c_2, example_ei_r_by_c_asym]


def test_ei_r_by_c_summary(two_r_by_c_ei_runs):  # pylint: disable=redefined-outer-name
    example_r_by_c_ei = two_r_by_c_ei_runs[0]  # pylint: disable=redefined-outer-name
    assert isinstance(example_r_by_c_ei.summary(), str)


# @TODO: this test fails on github but not locally - fix
# def test_io_utils(two_r_by_c_ei_runs):  # pylint: disable=redefined-outer-name
#     example_r_by_c_ei = two_r_by_c_ei_runs[0]  # pylint: disable=redefined-outer-name
#     model_name_orig = example_r_by_c_ei.model_name
#     to_netcdf(example_r_by_c_ei, "example.nc")
#     reloaded_ei = from_netcdf("example.nc")
#     assert isinstance(reloaded_ei.summary(), str)  # check that summary string is there
#     assert model_name_orig == reloaded_ei.model_name  # check that model data came with


def test_ei_calculate_turnout_adjusted_samples(
    two_r_by_c_ei_runs,
):  # pylint: disable=redefined-outer-name
    def calculate_turnout_adjust_samples_basic(
        non_adjusted_samples, abstain_column_name, candidate_names
    ):
        # non_adjusted_samples = self.sim_trace.get_values("b") num_samples x num_precincts x r x c
        num_samples, num_precincts, num_rows, _ = non_adjusted_samples.shape

        abstain_column_index = candidate_names.index(abstain_column_name)

        turnout_adjusted_samples = np.delete(non_adjusted_samples, abstain_column_index, axis=3)

        for r_idx in range(num_rows):
            for samp_idx in range(num_samples):
                for p_idx in range(num_precincts):
                    # for c_idx in range(c-1):
                    turnout_adjusted_samples[samp_idx, p_idx, r_idx, :] = (
                        turnout_adjusted_samples[samp_idx, p_idx, r_idx, :]
                        / turnout_adjusted_samples[samp_idx, p_idx, r_idx, :].sum()
                    )

        return turnout_adjusted_samples

    example_r_by_c_ei = two_r_by_c_ei_runs[0]  # pylint: disable=redefined-outer-name
    non_adjusted_samples = np.transpose(
        example_r_by_c_ei.sim_trace["posterior"]["b"].stack(all_draws=["chain", "draw"]).values,
        axes=(3, 0, 1, 2),
    )
    test_adj_samples = calculate_turnout_adjust_samples_basic(
        non_adjusted_samples, "Hardy", example_r_by_c_ei.candidate_names
    )

    example_r_by_c_ei.calculate_turnout_adjusted_summary(["Hardy"])
    turnout_adjusted_samps_pyei = example_r_by_c_ei.turnout_adjusted_samples
    assert np.all(np.isclose(test_adj_samples, turnout_adjusted_samps_pyei))


def test_computation_of_districtwide_samples(
    two_r_by_c_ei_runs,
):  # pylint: disable=redefined-outer-name:  # pylint: disable=redefined-outer-name:
    """
    Testing calculations of RowByColumnEI.sampled_voting_prefs
    """
    ei_ex = two_r_by_c_ei_runs[0]

    def calculate_districtwide_samples_basic(samples, demographic_group_fractions, precinct_pops):
        # group fraction has shape: r x num_precincts
        demographic_group_counts = np.transpose(
            demographic_group_fractions * precinct_pops
        )  # num_precincts x r
        num_samples, _, num_rows, num_cols = samples.shape
        districtwide_prefs = np.empty((num_samples, num_rows, num_cols))
        for r_idx in range(num_rows):
            for samp_idx in range(num_samples):
                for c_idx in range(num_cols):
                    districtwide_prefs[samp_idx, r_idx, c_idx] = (
                        samples[samp_idx, :, r_idx, c_idx] * demographic_group_counts[:, r_idx]
                    ).sum() / (demographic_group_counts[:, r_idx]).sum()
        return districtwide_prefs

    test_districtwide_prefs = calculate_districtwide_samples_basic(
        np.transpose(
            ei_ex.sim_trace["posterior"]["b"].stack(all_draws=["chain", "draw"]).values,
            axes=(3, 0, 1, 2),
        ),
        ei_ex.demographic_group_fractions,
        ei_ex.precinct_pops,
    )
    np.all(np.isclose(test_districtwide_prefs, ei_ex.sampled_voting_prefs))


def test_candidate_of_choice_report(
    two_r_by_c_ei_runs,
):  # pylint: disable=redefined-outer-name
    example_r_by_c_ei = two_r_by_c_ei_runs[0]  # pylint: disable=redefined-outer-name
    candidate_preference_rate_dict = example_r_by_c_ei.candidate_of_choice_report(
        verbose=True, non_candidate_names=None
    )
    assert 0 <= candidate_preference_rate_dict["e_asian", "Kolstad"] < 1


def test_candidate_of_choice_polarization_report(
    two_r_by_c_ei_runs,
):  # pylint: disable=redefined-outer-name)
    example_r_by_c_ei = two_r_by_c_ei_runs[0]  # pylint: disable=redefined-outer-name
    candidate_differ_rate_dict = example_r_by_c_ei.candidate_of_choice_polarization_report(
        verbose=True, non_candidate_names=None
    )
    assert candidate_differ_rate_dict["non_asian", "e_asian"] > 0.4


def test_polarization_report(
    two_r_by_c_ei_runs,
):  # pylint: disable=redefined-outer-name
    example_r_by_c_ei = two_r_by_c_ei_runs[0]  # pylint: disable=redefined-outer-name
    groups = ["e_asian", "non_asian"]
    candidate = "Kolstad"
    prob_20 = example_r_by_c_ei.polarization_report(groups, candidate, threshold=0.2)
    prob_40 = example_r_by_c_ei.polarization_report(groups, candidate, threshold=0.4)
    thresh_95_range = example_r_by_c_ei.polarization_report(groups, candidate, percentile=95)
    thresh_90_range = example_r_by_c_ei.polarization_report(groups, candidate, percentile=90)

    assert prob_20 >= prob_40
    assert thresh_95_range[1] > thresh_95_range[0]
    assert thresh_95_range[1] - thresh_95_range[0] >= thresh_90_range[1] - thresh_90_range[0]


# TEST PLOTTING


def test_ei_r_by_c_precinct_scatterplot(
    two_r_by_c_ei_runs,
):  # pylint: disable=redefined-outer-name
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


def test_plot_polarization_kde(
    two_r_by_c_ei_runs,
):  # pylint: disable=redefined-outer-name
    example_r_by_c_ei = two_r_by_c_ei_runs[0]  # pylint: disable=redefined-outer-name
    groups = ["ind", "e_asian"]
    candidate = "Kolstad"
    percentile_ax = example_r_by_c_ei.plot_polarization_kde(
        groups, candidate, threshold=0.4, show_threshold=True
    )
    percentile_ax_2 = example_r_by_c_ei.plot_polarization_kde(
        groups, candidate, threshold=0.4, show_threshold=True
    )
    threshold_ax = example_r_by_c_ei.plot_polarization_kde(
        groups, candidate, percentile=95, show_threshold=True
    )
    assert percentile_ax is not None
    print("done 1")
    assert percentile_ax_2 is not None
    print("done 2")
    assert threshold_ax is not None
    print("done 3")
    with pytest.raises(ValueError):
        example_r_by_c_ei.plot_polarization_kde(
            groups, candidate, threshold=0.4, percentile=95, show_threshold=True
        )
    print("done")


def test_plot_margin_kde(two_r_by_c_ei_runs):  # pylint: disable=redefined-outer-name
    example_r_by_c_ei = two_r_by_c_ei_runs[0]  # pylint: disable=redefined-outer-name
    example_r_by_c_ei.plot_margin_kde("ind", ["Hardy", "Nadeem"], threshold=0.1)
