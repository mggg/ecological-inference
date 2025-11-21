"""Tests for data.py"""

import pytest

from pyei import data


@pytest.fixture
def expected_data_sets():  # noqa: ANN201
    """Hardcode the names of existing datasets.

    We could also use `dir` or `vars`, but this is quicker for now...
    """
    return ("Santa_Clara", "Waterbury")


def test_data_sets_exist(expected_data_sets):  # noqa: ANN001,ANN201,D103
    for data_set in expected_data_sets:  # noqa: ANN001
        assert hasattr(data.Datasets, data_set)
