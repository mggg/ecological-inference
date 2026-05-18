"""Tests for save/load roundtrips through pyei.io_utils."""

import numpy as np
import pytest

from pyei.io_utils import from_netcdf, to_netcdf
from pyei.r_by_c import RowByColumnEI
from pyei.two_by_two import TwoByTwoEI

# Every roundtrip needs a fitted EI; the unfit-error test is the lone
# fast case and we accept the mislabel rather than complicate the markup.
pytestmark = pytest.mark.slow


@pytest.fixture(scope="session")
def r_by_c_roundtrip_file(two_r_by_c_ei_runs, tmp_path_factory):
    """Roundtrip a fitted RowByColumnEI through netCDF and return both ends."""
    original = two_r_by_c_ei_runs[0]
    target = tmp_path_factory.mktemp("io") / "rxc.nc"
    to_netcdf(original, str(target))
    reloaded = from_netcdf(str(target))
    return original, reloaded


@pytest.fixture(scope="session")
def two_by_two_roundtrip_file(example_two_by_two_ei, tmp_path_factory):
    """Roundtrip a fitted TwoByTwoEI through netCDF and return both ends."""
    original = example_two_by_two_ei
    target = tmp_path_factory.mktemp("io") / "twobytwo.nc"
    to_netcdf(original, str(target))
    reloaded = from_netcdf(str(target))
    return original, reloaded


def test_to_netcdf_writes_file(two_r_by_c_ei_runs, tmp_path):
    target = tmp_path / "out.nc"
    to_netcdf(two_r_by_c_ei_runs[0], str(target))
    assert target.exists()
    assert target.stat().st_size > 0


def test_to_netcdf_raises_when_not_fit(tmp_path):
    ei = RowByColumnEI(model_name="multinomial-dirichlet")
    with pytest.raises(ValueError, match="must be fit"):
        to_netcdf(ei, str(tmp_path / "unfit.nc"))


def test_r_by_c_roundtrip_preserves_model_name(r_by_c_roundtrip_file):
    original, reloaded = r_by_c_roundtrip_file
    assert isinstance(reloaded, RowByColumnEI)
    assert reloaded.model_name == original.model_name


def test_r_by_c_roundtrip_preserves_group_and_candidate_names(r_by_c_roundtrip_file):
    original, reloaded = r_by_c_roundtrip_file
    assert list(reloaded.demographic_group_names) == list(
        original.demographic_group_names
    )
    assert list(reloaded.candidate_names) == list(original.candidate_names)


def test_r_by_c_roundtrip_preserves_fractions(r_by_c_roundtrip_file):
    original, reloaded = r_by_c_roundtrip_file
    np.testing.assert_array_equal(
        reloaded.demographic_group_fractions, original.demographic_group_fractions
    )
    np.testing.assert_array_equal(reloaded.votes_fractions, original.votes_fractions)


def test_r_by_c_roundtrip_preserves_posterior_samples(r_by_c_roundtrip_file):
    original, reloaded = r_by_c_roundtrip_file
    orig_b = original.sim_trace["posterior"]["b"].values
    reloaded_b = reloaded.sim_trace["posterior"]["b"].values
    np.testing.assert_allclose(reloaded_b, orig_b)


def test_r_by_c_roundtrip_summary_renders(r_by_c_roundtrip_file):
    _, reloaded = r_by_c_roundtrip_file
    summary = reloaded.summary()
    assert isinstance(summary, str)
    assert "Hardy" in summary  # candidate name survives


def test_two_by_two_roundtrip_preserves_model_name(two_by_two_roundtrip_file):
    original, reloaded = two_by_two_roundtrip_file
    assert isinstance(reloaded, TwoByTwoEI)
    assert reloaded.model_name == original.model_name


def test_two_by_two_roundtrip_preserves_posterior(two_by_two_roundtrip_file):
    original, reloaded = two_by_two_roundtrip_file
    orig_b = original.sim_trace["posterior"]["b_1"].values
    reloaded_b = reloaded.sim_trace["posterior"]["b_1"].values
    np.testing.assert_allclose(reloaded_b, orig_b)
