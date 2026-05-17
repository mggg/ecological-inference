"""Test plot utils."""

import pytest

from pyei.plot_utils import tomography_plot


def test_tomography_plot(example_two_by_two_data):
    tomography_plot(
        example_two_by_two_data["group_fractions"],
        example_two_by_two_data["votes_fractions"],
        example_two_by_two_data["demographic_group_name"],
        example_two_by_two_data["candidate_name"],
    )


def test_ei_plot_and_plot_summary(example_two_by_two_ei):
    axes = example_two_by_two_ei.plot()
    assert len(axes) == 2  # one axis each for boxplot and kde


def test_ei_plot_kdes(example_two_by_two_ei):
    ax = example_two_by_two_ei.plot_kde()
    assert ax is not None


def test_ei_plot_boxplot(example_two_by_two_ei):
    ax = example_two_by_two_ei.plot_boxplot()
    assert ax is not None


def test_ei_plot_intervals(example_two_by_two_ei):
    ax = example_two_by_two_ei.plot_intervals()
    assert ax is not None


def test_ei_precinct_level_plots(example_two_by_two_ei):
    ax = example_two_by_two_ei.precinct_level_plot()
    assert ax is not None


def test_ei_plot_intervals_by_precinct(example_two_by_two_ei):
    ax = example_two_by_two_ei.plot_intervals_by_precinct()
    assert ax is not None


def test_plot_polarization_kde(example_two_by_two_ei):
    percentile_ax = example_two_by_two_ei.plot_polarization_kde(
        threshold=0.4, show_threshold=True
    )
    percentile_ax_2 = example_two_by_two_ei.plot_polarization_kde(
        threshold=0.4, reference_group=1, show_threshold=True
    )
    threshold_ax = example_two_by_two_ei.plot_polarization_kde(
        percentile=95, show_threshold=True
    )
    assert percentile_ax is not None
    assert percentile_ax_2 is not None
    assert threshold_ax is not None
    with pytest.raises(ValueError):
        example_two_by_two_ei.plot_polarization_kde(
            threshold=0.4, percentile=95, show_threshold=True
        )
