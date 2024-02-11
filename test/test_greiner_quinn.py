"""Test Greiner Quinn Gibbs sampler."""

# pylint: disable=duplicate-code
import random
import pytest
import numpy as np
import scipy.stats as st


from pyei import data
from pyei.r_by_c import RowByColumnEI
from pyei.greiner_quinn_gibbs_sampling import (
    get_initial_internal_count_sample,
    theta_to_omega,
    greiner_quinn_gibbs_sample,
)
from pyei.distribution_utils import non_central_hypergeometric_sample


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
    group_counts = np.round(group_fractions * precinct_pops)
    vote_counts = np.round(votes_fractions * precinct_pops)
    num_groups = group_counts.shape[0]
    num_candidates = vote_counts.shape[0]

    group_diff = group_counts.sum(axis=0) - precinct_pops
    for idx_of_mismatch in np.where(group_diff != 0):
        group_to_adjust = random.randint(0, num_groups - 1)
        group_counts[group_to_adjust, idx_of_mismatch] -= group_diff[idx_of_mismatch]

    vote_diff = vote_counts.sum(axis=0) - precinct_pops
    for idx_of_mismatch in np.where(vote_diff != 0):
        candidate_to_adjust = random.randint(0, num_candidates - 1)
        vote_counts[candidate_to_adjust, idx_of_mismatch] -= vote_diff[idx_of_mismatch]

    group_counts = group_counts.T
    vote_counts = vote_counts.T
    return {
        "group_fractions": group_fractions,
        "votes_fractions": votes_fractions,
        "group_counts": group_counts,
        "vote_counts": vote_counts,
        "precinct_pops": precinct_pops,
        "demographic_group_names": demographic_group_names,
        "candidate_names": candidate_names,
    }


def test_get_initial_internal_count_sample(
    example_r_by_c_data_asym,
):  # pylint: disable=redefined-outer-name:
    vote_counts = example_r_by_c_data_asym["vote_counts"]
    group_counts = example_r_by_c_data_asym["group_counts"]
    precinct_pops = example_r_by_c_data_asym["precinct_pops"]
    samp = get_initial_internal_count_sample(group_counts, vote_counts, precinct_pops)
    samp_py = get_initial_internal_count_sample.py_func(group_counts, vote_counts, precinct_pops)

    assert np.all(samp.sum(axis=2) - group_counts == 0)  # sample respects given group counts
    assert np.all(samp.sum(axis=1) - vote_counts == 0)  # sample respects given vote counts
    assert np.all(samp_py.sum(axis=2) - group_counts == 0)  # sample respects given group counts
    assert np.all(samp_py.sum(axis=1) - vote_counts == 0)  # sample respects given vote counts


def test_theta_to_omega():
    num_precincts = 8
    r = 3
    c = 4
    alpha = np.ones(c)
    theta = st.dirichlet.rvs(alpha, size=(num_precincts, r))
    omega = theta_to_omega(theta)
    assert omega.shape[2] == c - 1
    np.testing.assert_almost_equal(
        omega[3, 2, 1], np.log(theta[3, 2, 1] / theta[3, 2, c - 1]), decimal=4
    )


def test_greiner_quinn_gibbs_sample(
    example_r_by_c_data_asym,
):  # pylint: disable=redefined-outer-name
    r = example_r_by_c_data_asym["group_counts"].shape[1]
    c = example_r_by_c_data_asym["vote_counts"].shape[1]
    print(r, c)
    num_samples = 100
    nu_0 = 10
    psi_0 = 1 / 10 * np.identity(r * (c - 1))  # relates to prior precision
    k_0_inv = 1 / (0.5) * np.identity(r * (c - 1))  # same size as Sigma

    votes_all_precincts = example_r_by_c_data_asym["vote_counts"].sum(axis=0)
    log_ratio_support = np.log(
        votes_all_precincts[0:-1] / votes_all_precincts[-1]
    )  # assume each group supports according to the total vote split
    mu_0 = np.tile(log_ratio_support, r)  # shape r * (c-1)
    gamma = 0.1

    greiner_quinn_gibbs_sample(
        example_r_by_c_data_asym["group_counts"],
        example_r_by_c_data_asym["vote_counts"],
        num_samples,
        nu_0,
        psi_0,
        k_0_inv,
        mu_0,
        gamma=gamma,
    )


def test_pyei_greiner_quinn_gibbs(
    example_r_by_c_data_asym,
):  # pylint: disable=redefined-outer-name
    ei_greiner_quinn = RowByColumnEI(model_name="greiner-quinn")
    ei_greiner_quinn.fit(
        example_r_by_c_data_asym["group_fractions"],
        example_r_by_c_data_asym["votes_fractions"],
        example_r_by_c_data_asym["precinct_pops"],
        num_samples=5,
        burnin=1,
    )


def test_non_central_hypergeometric_sample():
    samp = non_central_hypergeometric_sample.py_func(10, 5, 7, 1)
    assert samp >= 2
    assert samp <= 10
    samp2 = non_central_hypergeometric_sample.py_func(10, 10, 7, 1)
    assert samp2 <= 10
