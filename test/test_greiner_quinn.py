"""Test Greiner Quinn Gibbs sampler."""

# pylint: disable=duplicate-code
import numpy as np
import scipy.stats as st

from pyei.distribution_utils import non_central_hypergeometric_sample
from pyei.greiner_quinn_gibbs_sampling import (
    _get_initial_internal_count_sample,
    _theta_to_omega,
    greiner_quinn_gibbs_sample,
)
from pyei.r_by_c import RowByColumnEI


def test_get_initial_internal_count_sample(example_r_by_c_data_asym):
    vote_counts = example_r_by_c_data_asym["vote_counts"]
    group_counts = example_r_by_c_data_asym["group_counts"]
    precinct_pops = example_r_by_c_data_asym["precinct_pops"]
    samp = _get_initial_internal_count_sample(group_counts, vote_counts, precinct_pops)
    samp_py = _get_initial_internal_count_sample.py_func(
        group_counts, vote_counts, precinct_pops
    )

    assert np.all(samp.sum(axis=2) - group_counts == 0)
    assert np.all(samp.sum(axis=1) - vote_counts == 0)
    assert np.all(samp_py.sum(axis=2) - group_counts == 0)
    assert np.all(samp_py.sum(axis=1) - vote_counts == 0)


def test_theta_to_omega():
    num_precincts = 8
    r = 3
    c = 4
    alpha = np.ones(c)
    theta = st.dirichlet.rvs(alpha, size=(num_precincts, r))
    omega = _theta_to_omega(theta)
    assert omega.shape[2] == c - 1
    np.testing.assert_almost_equal(
        omega[3, 2, 1], np.log(theta[3, 2, 1] / theta[3, 2, c - 1]), decimal=4
    )


def test_greiner_quinn_gibbs_sample(example_r_by_c_data_asym):
    r = example_r_by_c_data_asym["group_counts"].shape[1]
    c = example_r_by_c_data_asym["vote_counts"].shape[1]
    num_samples = 100
    nu_0 = 10
    psi_0 = 1 / 10 * np.identity(r * (c - 1))
    k_0_inv = 1 / (0.5) * np.identity(r * (c - 1))

    votes_all_precincts = example_r_by_c_data_asym["vote_counts"].sum(axis=0)
    log_ratio_support = np.log(votes_all_precincts[0:-1] / votes_all_precincts[-1])
    mu_0 = np.tile(log_ratio_support, r)
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


def test_pyei_greiner_quinn_gibbs(example_r_by_c_data_asym):
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
