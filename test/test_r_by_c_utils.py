"""Tests for the dimension validator used by RowByColumnEI.fit.

``check_dimensions_of_input`` is the only guard between user input and the
PyMC model. Each error branch should fail loudly with a typed exception
rather than letting a mis-shaped array reach the sampler.
"""

import numpy as np
import pytest

from pyei.r_by_c_utils import check_dimensions_of_input


def _well_formed_inputs(num_precincts=5, r=2, c=3):
    """Construct a self-consistent set of inputs."""
    group_fractions = np.full((r, num_precincts), 1.0 / r)
    votes_fractions = np.full((c, num_precincts), 1.0 / c)
    precinct_pops = np.full(num_precincts, 100, dtype=np.int64)
    return group_fractions, votes_fractions, precinct_pops, [r, c]


def test_happy_path_no_warning_no_error():
    group_fractions, votes_fractions, precinct_pops, dims = _well_formed_inputs()
    # filterwarnings = ["error"] in pyproject promotes any unexpected warning
    # to a failure, so an unannotated call is the actual assertion.
    check_dimensions_of_input(
        group_fractions,
        votes_fractions,
        precinct_pops,
        ["a", "b"],
        ["c1", "c2", "c3"],
        dims,
    )


def test_votes_fractions_precinct_mismatch_raises():
    group_fractions, votes_fractions, precinct_pops, dims = _well_formed_inputs()
    bad_votes = votes_fractions[:, :-1]  # one fewer precinct
    with pytest.raises(
        ValueError, match="votes_fractions should have shape: c x num_precincts"
    ):
        check_dimensions_of_input(
            group_fractions, bad_votes, precinct_pops, None, None, dims
        )


def test_group_fractions_precinct_mismatch_raises():
    group_fractions, votes_fractions, precinct_pops, dims = _well_formed_inputs()
    bad_groups = group_fractions[:, :-1]
    # Drop votes_fractions to the same precinct count so the first guard
    # passes and we exercise the second one.
    matched_votes = votes_fractions[:, :-1]
    with pytest.raises(ValueError):
        check_dimensions_of_input(
            bad_groups,
            matched_votes,
            precinct_pops,
            None,
            None,
            dims,
        )


def test_wrong_demographic_group_names_length_warns():
    group_fractions, votes_fractions, precinct_pops, dims = _well_formed_inputs(r=2)
    with pytest.warns(UserWarning, match="demographic_groups_names"):
        check_dimensions_of_input(
            group_fractions,
            votes_fractions,
            precinct_pops,
            ["only_one"],  # r=2 but only one name
            None,
            dims,
        )


def test_wrong_candidate_names_length_warns():
    group_fractions, votes_fractions, precinct_pops, dims = _well_formed_inputs(c=3)
    with pytest.warns(UserWarning, match="candidate_names"):
        check_dimensions_of_input(
            group_fractions,
            votes_fractions,
            precinct_pops,
            None,
            ["c1", "c2"],  # c=3 but only two names
            dims,
        )
