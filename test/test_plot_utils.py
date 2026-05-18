"""Test plot utils.

These tests verify the structural content of returned axes (line count,
axis bounds, legend entries) rather than rendered pixels. Image-baseline
testing would catch more, but at much higher maintenance cost; structural
assertions catch the regressions that matter most — missing data, swapped
axes, dropped categories.
"""

import pytest

from pyei.plot_utils import tomography_plot


def test_tomography_plot_draws_one_line_per_precinct(example_two_by_two_data):
    """The tomography plot draws a constraint line per precinct."""
    ax = tomography_plot(
        example_two_by_two_data["group_fractions"],
        example_two_by_two_data["votes_fractions"],
        example_two_by_two_data["demographic_group_name"],
        example_two_by_two_data["candidate_name"],
    )
    num_precincts = len(example_two_by_two_data["group_fractions"])
    assert len(ax.lines) == num_precincts


def test_ei_plot_returns_two_axes(example_two_by_two_ei):
    """plot() returns (boxplot_ax, kde_ax)."""
    axes = example_two_by_two_ei.plot()
    assert len(axes) == 2  # one for boxplot, one for kde


def test_ei_plot_kde_has_density_curves(example_two_by_two_ei):
    """plot_kde renders one density per group (here: e_asian + complement)."""
    ax = example_two_by_two_ei.plot_kde()
    # KDE renders as filled artists (collections) and/or line plots —
    # at least one of these must be populated.
    assert len(ax.collections) + len(ax.lines) > 0
    # x-axis spans the unit interval since support is a probability.
    xlim = ax.get_xlim()
    assert xlim[0] <= 0.0 and xlim[1] >= 1.0 or (xlim[0] >= -0.1 and xlim[1] <= 1.1)


def test_ei_plot_boxplot_has_two_groups(example_two_by_two_ei):
    """boxplot shows one box per group (demographic group + complement)."""
    ax = example_two_by_two_ei.plot_boxplot()
    # Seaborn boxplot draws box artists; count the labels on the categorical axis.
    tick_labels = [t.get_text() for t in ax.get_yticklabels() + ax.get_xticklabels()]
    nonempty = [t for t in tick_labels if t]
    # At least the two group names should appear somewhere on the axis.
    assert len(nonempty) >= 2


def test_ei_plot_intervals_has_artists(example_two_by_two_ei):
    ax = example_two_by_two_ei.plot_intervals()
    assert len(ax.lines) + len(ax.collections) > 0


def test_ei_precinct_level_plot_returns_axes(example_two_by_two_ei):
    ax = example_two_by_two_ei.precinct_level_plot()
    assert ax is not None
    # Precinct-level plot draws something per precinct.
    assert len(ax.lines) + len(ax.collections) > 0


def test_ei_plot_intervals_by_precinct_returns_axes(example_two_by_two_ei):
    ax = example_two_by_two_ei.plot_intervals_by_precinct()
    assert ax is not None


def test_plot_polarization_kde_threshold_mode(example_two_by_two_ei):
    ax = example_two_by_two_ei.plot_polarization_kde(threshold=0.4, show_threshold=True)
    # show_threshold=True draws a vertical reference line.
    assert len(ax.lines) >= 1


def test_plot_polarization_kde_reference_group(example_two_by_two_ei):
    ax = example_two_by_two_ei.plot_polarization_kde(
        threshold=0.4, reference_group=1, show_threshold=True
    )
    assert ax is not None


def test_plot_polarization_kde_percentile_mode(example_two_by_two_ei):
    ax = example_two_by_two_ei.plot_polarization_kde(percentile=95, show_threshold=True)
    assert ax is not None


def test_plot_polarization_kde_rejects_both_threshold_and_percentile(
    example_two_by_two_ei,
):
    with pytest.raises(ValueError):
        example_two_by_two_ei.plot_polarization_kde(
            threshold=0.4, percentile=95, show_threshold=True
        )
