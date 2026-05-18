"""Tests for save/load roundtrips through pyei.io_utils."""

import numpy as np
import pytest

from pyei.goodmans_er import GoodmansERBayes
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


def test_r_by_c_roundtrip_preserves_precinct_pops(r_by_c_roundtrip_file):
    original, reloaded = r_by_c_roundtrip_file
    np.testing.assert_array_equal(reloaded.precinct_pops, original.precinct_pops)


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


def test_two_by_two_roundtrip_preserves_precinct_pops(two_by_two_roundtrip_file):
    original, reloaded = two_by_two_roundtrip_file
    np.testing.assert_array_equal(reloaded.precinct_pops, original.precinct_pops)


def test_two_by_two_roundtrip_preserves_group_and_candidate_names(
    two_by_two_roundtrip_file,
):
    original, reloaded = two_by_two_roundtrip_file
    assert reloaded.demographic_group_name == original.demographic_group_name
    assert reloaded.candidate_name == original.candidate_name


def test_two_by_two_roundtrip_recomputes_sampled_voting_prefs(
    two_by_two_roundtrip_file,
):
    # ``from_netcdf`` calls ``calculate_sampled_voting_prefs`` for 2x2; the
    # reload must reproduce both elements of the prefs pair within numerical
    # tolerance of the original.
    original, reloaded = two_by_two_roundtrip_file
    for orig_arr, reloaded_arr in zip(
        original.sampled_voting_prefs, reloaded.sampled_voting_prefs, strict=True
    ):
        assert orig_arr is not None and reloaded_arr is not None
        np.testing.assert_allclose(reloaded_arr, orig_arr)


# ---------------------------------------------------------------------------
# GoodmansERBayes round-trip: not supported, but the failure must surface
# at load time with a clear message rather than as a shape-broadcast crash.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def fitted_goodmans_er_bayes(example_two_by_two_data):
    """Fit a ``GoodmansERBayes`` for the round-trip tests below.

    ``cores=1`` is required so PyMC doesn't ``os.fork()`` after numpyro
    (used by upstream r-by-c fits) has initialised JAX — same constraint
    as ``test_two_by_two_fit_truncated_normal_produces_valid_output``.
    """
    ex = example_two_by_two_data
    ei = GoodmansERBayes("goodman_er_bayes", weighted_by_pop=True, sigma=1)
    ei.fit(
        ex["group_fractions"],
        ex["votes_fractions"],
        ex["precint_pops"],
        demographic_group_name=ex["demographic_group_name"],
        candidate_name=ex["candidate_name"],
        tune=500,
        random_seed=0,
        cores=1,
    )
    return ei


def test_goodmans_er_bayes_to_netcdf_succeeds(fitted_goodmans_er_bayes, tmp_path):
    """Saving a ``GoodmansERBayes`` must not crash on missing attrs

    Regression for the AttributeError on ``getattr(ei_object, attr)`` —
    ``to_netcdf`` now uses ``getattr(ei_object, attr, None)`` so missing
    attrs are treated the same as explicit None.
    """
    target = tmp_path / "goodman_er_bayes.nc"
    to_netcdf(fitted_goodmans_er_bayes, str(target))
    assert target.exists()
    assert target.stat().st_size > 0


def test_goodmans_er_bayes_from_netcdf_raises_clearly(
    fitted_goodmans_er_bayes, tmp_path
):
    # ``from_netcdf`` hardcodes ``TwoByTwoEI`` reconstruction, and
    # ``TwoByTwoEI.calculate_sampled_voting_prefs`` doesn't match Goodman's district-level
    # (rather than per-precinct) b_1/b_2 posterior layout.
    target = tmp_path / "goodman_er_bayes.nc"
    to_netcdf(fitted_goodmans_er_bayes, str(target))
    with pytest.raises(NotImplementedError, match="GoodmansERBayes"):
        from_netcdf(str(target))
