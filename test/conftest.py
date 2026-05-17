"""Shared fixtures for the pyei test suite."""

import random

import numpy as np
import pytest

from pyei import data
from pyei.r_by_c import RowByColumnEI
from pyei.two_by_two import TwoByTwoEI


@pytest.fixture(autouse=True)
def _seed_rngs():
    """Seed stdlib and numpy RNGs before every test for determinism.

    Tests that draw random inputs (parameter generation, dataset
    perturbation) must not depend on uncontrolled state. Sampler-level
    seeds (PyMC, numpyro) are pinned separately at fit time.
    """
    random.seed(0)
    np.random.seed(0)


@pytest.fixture(scope="session")
def example_two_by_two_data():
    """Santa Clara dataset projected to a single group/candidate pair."""
    sc_data = data.Datasets.Santa_Clara.to_dataframe()
    group_fractions = np.array(sc_data["pct_e_asian_vote"])
    votes_fractions = np.array(sc_data["pct_for_hardy2"])
    precinct_pops = np.array(sc_data["total2"])
    return {
        "group_fractions": group_fractions,
        "votes_fractions": votes_fractions,
        "precint_pops": precinct_pops,
        "demographic_group_name": "e_asian",
        "candidate_name": "Hardy",
        "precinct_names": sc_data["precinct"],
    }


@pytest.fixture(scope="session")
def example_two_by_two_ei(example_two_by_two_data):
    """A fitted TwoByTwoEI instance for plotting/reporting tests."""
    ei_ex = TwoByTwoEI(
        model_name="king99_pareto_modification", pareto_scale=8, pareto_shape=2
    )
    ei_ex.fit(
        example_two_by_two_data["group_fractions"],
        example_two_by_two_data["votes_fractions"],
        example_two_by_two_data["precint_pops"],
        demographic_group_name=example_two_by_two_data["demographic_group_name"],
        candidate_name=example_two_by_two_data["candidate_name"],
        precinct_names=example_two_by_two_data["precinct_names"],
        draws=100,
        tune=100,
    )
    return ei_ex


@pytest.fixture(scope="session")
def example_r_by_c_data():
    """Trimmed Santa Clara dataset (3 groups, 3 candidates, 10 precincts)."""
    sc_data = data.Datasets.Santa_Clara.to_dataframe().iloc[:10, :]
    precinct_pops = np.array(sc_data["total2"])
    votes_fractions = np.array(
        sc_data[["pct_for_hardy2", "pct_for_kolstad2", "pct_for_nadeem2"]]
    ).T
    group_fractions = np.array(
        sc_data[["pct_ind_vote", "pct_e_asian_vote", "pct_non_asian_vote"]]
    ).T
    return {
        "group_fractions": group_fractions,
        "votes_fractions": votes_fractions,
        "precinct_pops": precinct_pops,
        "demographic_group_names": ["ind", "e_asian", "non_asian"],
        "candidate_names": ["Hardy", "Kolstad", "Nadeem"],
    }


@pytest.fixture(scope="session")
def example_r_by_c_data_asym():
    """Trimmed Santa Clara dataset with r != c (2 groups, 3 candidates).

    Group and vote count marginals are adjusted to agree with precinct
    populations. We seed inside the fixture because the function-scoped
    autouse seeder fires after session fixtures are built.
    """
    random.seed(0)
    sc_data = data.Datasets.Santa_Clara.to_dataframe().iloc[:10, :]
    precinct_pops = np.array(sc_data["total2"])
    votes_fractions = np.array(
        sc_data[["pct_for_hardy2", "pct_for_kolstad2", "pct_for_nadeem2"]]
    ).T
    group_fractions = np.array(sc_data[["pct_asian_vote", "pct_non_asian_vote"]]).T

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

    return {
        "group_fractions": group_fractions,
        "votes_fractions": votes_fractions,
        "group_counts": group_counts.T,
        "vote_counts": vote_counts.T,
        "precinct_pops": precinct_pops,
        "demographic_group_names": ["asian", "non_asian"],
        "candidate_names": ["Hardy", "Kolstad", "Nadeem"],
    }


def _fit_r_by_c(example_r_by_c_data, model_name):
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
def two_r_by_c_ei_runs(example_r_by_c_data, example_r_by_c_data_asym):
    """Two symmetric fits (different models) plus one asymmetric fit."""
    return [
        _fit_r_by_c(example_r_by_c_data, "multinomial-dirichlet"),
        _fit_r_by_c(example_r_by_c_data, "multinomial-dirichlet-modified"),
        _fit_r_by_c(example_r_by_c_data_asym, "multinomial-dirichlet"),
    ]
