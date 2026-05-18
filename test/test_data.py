"""Tests for the bundled dataset loaders.

Both datasets are fetched from external URLs at load time. We don't add a
``network`` marker because session-scoped fixtures elsewhere in the suite
already require the same fetch — these tests piggyback on that round trip.
"""

import numpy as np
import pandas as pd
import pytest

from pyei import data


@pytest.fixture(scope="session")
def santa_clara_df():
    return data.Datasets.Santa_Clara.to_dataframe()


@pytest.fixture(scope="session")
def waterbury_df():
    return data.Datasets.Waterbury.to_dataframe()


def test_datasets_class_exposes_known_attributes():
    assert hasattr(data.Datasets, "Santa_Clara")
    assert hasattr(data.Datasets, "Waterbury")


def test_santa_clara_shape(santa_clara_df):
    assert santa_clara_df.shape == (42, 21)


def test_santa_clara_columns_include_pyei_inputs(santa_clara_df):
    """Columns the rest of the test suite indexes by name must be present."""
    required = {
        "precinct",
        "total2",
        "pct_for_hardy2",
        "pct_for_kolstad2",
        "pct_for_nadeem2",
        "pct_asian_vote",
        "pct_non_asian_vote",
        "pct_e_asian_vote",
        "pct_ind_vote",
    }
    assert required.issubset(set(santa_clara_df.columns))


def test_santa_clara_fractions_in_unit_interval(santa_clara_df):
    """Every pct_* column should hold fractions in [0, 1]."""
    pct_cols = [c for c in santa_clara_df.columns if c.startswith("pct_")]
    for col in pct_cols:
        values = santa_clara_df[col].to_numpy()
        assert values.min() >= 0.0, f"{col} has negative entries"
        assert values.max() <= 1.0, f"{col} has entries above 1"


def test_santa_clara_complementary_fractions_sum_to_one(santa_clara_df):
    """asian + non_asian vote share is a binary partition of voters."""
    total = santa_clara_df["pct_asian_vote"] + santa_clara_df["pct_non_asian_vote"]
    np.testing.assert_allclose(total.to_numpy(), 1.0, atol=1e-6)


def test_santa_clara_candidate_fractions_sum_to_at_most_one(santa_clara_df):
    """The three candidate shares are mutually exclusive."""
    total = (
        santa_clara_df["pct_for_hardy2"]
        + santa_clara_df["pct_for_kolstad2"]
        + santa_clara_df["pct_for_nadeem2"]
    )
    assert total.max() <= 1.0 + 1e-6


def test_santa_clara_total2_is_positive_integer(santa_clara_df):
    total2 = santa_clara_df["total2"]
    assert pd.api.types.is_integer_dtype(total2)
    assert (total2 > 0).all()


def test_waterbury_shape(waterbury_df):
    assert waterbury_df.shape == (23, 7)


def test_waterbury_columns(waterbury_df):
    required = {
        "Precinct",
        "Total.Votes",
        "Tom.Foley",
        "Dan.Malloy",
        "White.Pct",
        "Black.Pct",
        "Hispanic.Pct",
    }
    assert required.issubset(set(waterbury_df.columns))


def test_waterbury_race_fractions_in_unit_interval(waterbury_df):
    """Each race column is a per-precinct fraction in [0, 1].

    Note: White/Black/Hispanic do NOT partition the population — Hispanic is
    an ethnicity orthogonal to race in the source data — so we don't assert
    a row-sum constraint, only per-column bounds.
    """
    for col in ["White.Pct", "Black.Pct", "Hispanic.Pct"]:
        values = waterbury_df[col].to_numpy()
        assert values.min() >= 0.0, f"{col} has negative entries"
        assert values.max() <= 1.0, f"{col} has entries above 1"
