"""Test Greiner Quinn Gibbs sampler."""

# pylint: disable=duplicate-code
import numpy as np
import pytest
import scipy.stats as st

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


@pytest.fixture(scope="module")
def gq_sample_run(example_r_by_c_data_asym):
    """One short Gibbs chain on the asymmetric fixture, shared across the
    structural / invariant tests below so we pay the sampling cost once."""
    r = example_r_by_c_data_asym["group_counts"].shape[1]
    c = example_r_by_c_data_asym["vote_counts"].shape[1]
    num_samples = 50
    burnin = 5
    nu_0 = 10
    psi_0 = 1 / 10 * np.identity(r * (c - 1))
    k_0_inv = 1 / (0.5) * np.identity(r * (c - 1))
    votes_all_precincts = example_r_by_c_data_asym["vote_counts"].sum(axis=0)
    log_ratio_support = np.log(votes_all_precincts[:-1] / votes_all_precincts[-1])
    mu_0 = np.tile(log_ratio_support, r)

    result = greiner_quinn_gibbs_sample(
        example_r_by_c_data_asym["group_counts"],
        example_r_by_c_data_asym["vote_counts"],
        num_samples,
        nu_0,
        psi_0,
        k_0_inv,
        mu_0,
        gamma=0.1,
        burnin=burnin,
    )
    return {
        "result": result,
        "r": r,
        "c": c,
        "num_kept": num_samples - burnin,
        "group_counts": example_r_by_c_data_asym["group_counts"],
        "vote_counts": example_r_by_c_data_asym["vote_counts"],
    }


@pytest.mark.slow
def test_greiner_quinn_gibbs_sample_returns_expected_keys(gq_sample_run):
    """The sampler returns a dict with theta / counts / mu / Sigma chains."""
    assert set(gq_sample_run["result"].keys()) == {"theta", "counts", "mu", "Sigma"}


@pytest.mark.slow
def test_greiner_quinn_gibbs_sample_shapes(gq_sample_run):
    """All four arrays are sized (num_samples - burnin, ...) along axis 0."""
    res = gq_sample_run["result"]
    r = gq_sample_run["r"]
    c = gq_sample_run["c"]
    num_kept = gq_sample_run["num_kept"]
    num_precincts = gq_sample_run["group_counts"].shape[0]
    assert res["theta"].shape == (num_kept, num_precincts, r, c)
    assert res["counts"].shape == (num_kept, num_precincts, r, c)
    assert res["mu"].shape == (num_kept, r * (c - 1))
    assert res["Sigma"].shape == (num_kept, r * (c - 1), r * (c - 1))


@pytest.mark.slow
def test_greiner_quinn_gibbs_sample_theta_is_simplex(gq_sample_run):
    """theta[sample, precinct, group, :] is a probability distribution over candidates."""
    theta = gq_sample_run["result"]["theta"]
    assert theta.min() >= 0.0
    assert theta.max() <= 1.0
    np.testing.assert_allclose(theta.sum(axis=-1), 1.0, atol=1e-9)


@pytest.mark.slow
def test_greiner_quinn_gibbs_sample_internal_counts_preserve_marginals(gq_sample_run):
    """Every sampled internal-count table must satisfy both marginal constraints.

    This is the load-bearing invariant of the inner sampler: a draw that
    violates the group or vote marginal would produce an invalid latent
    state and silently corrupt the chain.
    """
    counts = gq_sample_run["result"]["counts"]
    group_counts = gq_sample_run["group_counts"]
    vote_counts = gq_sample_run["vote_counts"]
    # sum across candidates → per-precinct group counts; equal for every sample.
    np.testing.assert_array_equal(
        counts.sum(axis=3),
        np.broadcast_to(group_counts, counts.shape[:-1]),
    )
    # sum across groups → per-precinct vote counts; equal for every sample.
    np.testing.assert_array_equal(
        counts.sum(axis=2),
        np.broadcast_to(vote_counts, (counts.shape[0], *vote_counts.shape)),
    )


@pytest.mark.slow
def test_greiner_quinn_gibbs_sample_sigma_is_positive_definite(gq_sample_run):
    """Every Sigma draw is an inverse-Wishart sample, so it must be SPD.

    Symmetry is exact (np.linalg.cholesky requires symmetric input); we
    check positive-definiteness via successful Cholesky on every sample.
    """
    sigma = gq_sample_run["result"]["Sigma"]
    for i in range(sigma.shape[0]):
        s = sigma[i]
        np.testing.assert_allclose(s, s.T, atol=1e-9)
        # cholesky raises LinAlgError if s is not positive definite.
        np.linalg.cholesky(s)


@pytest.mark.slow
def test_greiner_quinn_gibbs_sample_rejects_burnin_ge_num_samples(
    example_r_by_c_data_asym,
):
    r = example_r_by_c_data_asym["group_counts"].shape[1]
    c = example_r_by_c_data_asym["vote_counts"].shape[1]
    psi_0 = 1 / 10 * np.identity(r * (c - 1))
    k_0_inv = 1 / (0.5) * np.identity(r * (c - 1))
    mu_0 = np.zeros(r * (c - 1))
    with pytest.raises(ValueError, match="burn-in"):
        greiner_quinn_gibbs_sample(
            example_r_by_c_data_asym["group_counts"],
            example_r_by_c_data_asym["vote_counts"],
            num_samples=5,
            nu_0=10,
            psi_0=psi_0,
            k_0_inv=k_0_inv,
            mu_0=mu_0,
            burnin=5,
        )


@pytest.mark.slow
def test_pyei_greiner_quinn_gibbs_produces_valid_voting_prefs(example_r_by_c_data_asym):
    """The greiner-quinn model_name path must produce a usable RowByColumnEI:
    sampled_voting_prefs has the right shape, sums to 1 per group, and lies in [0, 1].
    """
    ei = RowByColumnEI(model_name="greiner-quinn")
    ei.fit(
        example_r_by_c_data_asym["group_fractions"],
        example_r_by_c_data_asym["votes_fractions"],
        example_r_by_c_data_asym["precinct_pops"],
        num_samples=20,
        burnin=2,
    )
    r, c = ei.num_groups_and_num_candidates
    assert ei.sampled_voting_prefs.shape[1:] == (r, c)
    assert ei.sampled_voting_prefs.min() >= 0.0
    assert ei.sampled_voting_prefs.max() <= 1.0
    np.testing.assert_allclose(ei.sampled_voting_prefs.sum(axis=2), 1.0, atol=1e-6)
