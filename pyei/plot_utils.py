"""Plotting functions for visualizing ei outputs"""
import warnings
import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import scipy.stats as st

__all__ = [
    "plot_precincts",
    "plot_boxplot",
    "plot_kdes",
    "plot_conf_or_credible_interval",
]


def plot_precincts(
    voting_prefs_group1, voting_prefs_group2, y_labels=None, show_all_precincts=False, ax=None
):
    """Ridgeplots of sampled voting preferences for each precinct"""
    overlap = 1.3
    if ax is None:
        _, ax = plt.subplots()
    x = np.linspace(0, 1, 500)  # 500 points between 0 and 1 on the x-axis

    N = voting_prefs_group1.shape[1]
    if N > 50 and not show_all_precincts:
        warnings.warn(
            f"User attempted to plot {N} precinct-level voting preference "
            f"ridgeplots. Automatically restricting to first 50 precincts "
            f"(run with `show_all_precincts=True` to plot all precinct ridgeplots.)"
        )
        voting_prefs_group1 = voting_prefs_group1[:, :50]
        voting_prefs_group2 = voting_prefs_group2[:, :50]
        if y_labels is not None:
            y_labels = y_labels[:50]
        N = 50
    if y_labels is None:
        y_labels = range(N)

    iterator = zip(y_labels, voting_prefs_group1.T, voting_prefs_group2.T)

    for idx, (precinct, group1, group2) in enumerate(iterator, 1):
        pfx = "" if idx == 1 else "_"
        group1_kde = st.gaussian_kde(group1)
        group2_kde = st.gaussian_kde(group2)
        ax.plot([0], [precinct])
        trans = ax.convert_yunits(precinct)

        group1_y = group1_kde(x)
        group1_y = overlap * group1_y / group1_y.max()
        group2_y = group2_kde(x)
        group2_y = overlap * group2_y / group2_y.max()

        ax.fill_between(
            x,
            group1_y + trans,
            trans,
            color="steelblue",
            zorder=4 * N - 4 * idx,
            label=pfx + "Group 1",
        )
        ax.plot(x, group1_y + trans, color="black", linewidth=1, zorder=4 * N - 4 * idx + 1)

        ax.fill_between(
            x,
            group2_y + trans,
            trans,
            color="orange",
            zorder=4 * N - 4 * idx + 2,
            label=pfx + "Group 2",
        )
        ax.plot(x, group2_y + trans, color="black", linewidth=1, zorder=4 * N - 4 * idx + 3)
    ax.set_title("Precinct level estimates of voting preferences")
    ax.set_xlabel("Percent vote for candidate")
    return ax


def plot_boxplot(voting_prefs_group1, voting_prefs_group2, group1_name, group2_name, ax=None):
    """
    Horizontal boxplot of 2 groups of samples between 0 and 1
    """
    if ax is None:
        ax = plt.gca()
    samples_df = pd.DataFrame({group1_name: voting_prefs_group1, group2_name: voting_prefs_group2})
    ax = sns.boxplot(data=samples_df, orient="h", ax=ax)
    ax.set_xlim((0, 1))
    return ax


def plot_summary(voting_prefs_group1, voting_prefs_group2, group1_name, group2_name):
    """ Plot KDE, histogram, and boxplot"""
    _, (ax_box, ax_hist) = plt.subplots(
        2, sharex=True, figsize=(12, 6.4), gridspec_kw={"height_ratios": (0.15, 0.85)}
    )
    sns.despine(ax=ax_hist)
    sns.despine(ax=ax_box, left=True)
    # plot custom boxplot, with two boxplots in the same row
    colors = sns.color_palette()  # fetch seaborn default color palette
    plot_props = dict(fliersize=5, linewidth=2, whis=[5, 95])
    flier1_props = dict(marker="o", markerfacecolor=colors[0], alpha=0.5)
    flier2_props = dict(marker="d", markerfacecolor=colors[1], alpha=0.5)
    sns.boxplot(
        voting_prefs_group1,
        orient="h",
        color=colors[0],
        ax=ax_box,
        flierprops=flier1_props,
        **plot_props,
    )
    sns.boxplot(
        voting_prefs_group2,
        orient="h",
        color=colors[1],
        ax=ax_box,
        flierprops=flier2_props,
        **plot_props,
    )
    ax_box.tick_params(axis="y", left=False)  # remove y axis ticks

    # plot distribution
    plot_kdes(voting_prefs_group1, voting_prefs_group2, group1_name, group2_name, ax=ax_hist)
    return (ax_box, ax_hist)


def plot_kdes(voting_prefs_group1, voting_prefs_group2, group1_name, group2_name, ax=None):
    """'
    Plot kernel density plots of samples between 0 and 1 (e.g. of voting preferences) for two groups
    """
    if ax is None:
        ax = plt.gca()
    ax.set_xlim((0, 1))
    sns.distplot(voting_prefs_group1, hist=True, ax=ax, label=group1_name)
    sns.distplot(voting_prefs_group2, hist=True, ax=ax, label=group2_name)
    ax.legend()
    return ax


def plot_conf_or_credible_interval(
    interval_1, interval_2, group1_name, group2_name, candidate_name, title, ax=None
):
    """
    Plot confidence of credible interval for two different groups
    """

    # TO DO: generalize for more intervals
    int1_height = 0.4
    int2_height = 0.2
    if ax is None:
        ax = plt.axes(frameon=False)

    ax.set(
        title=title,
        xlim=(0, 1),
        ylim=(0, 0.5),
        xlabel=f"Support for {candidate_name}",
        frame_on=False,
        aspect=0.3,
    )

    ax.get_xaxis().tick_bottom()
    ax.axes.get_yaxis().set_visible(False)
    ax.text(1, int1_height, group1_name)
    ax.text(1, int2_height, group2_name)
    ax.plot(interval_1, [int1_height, int1_height], linewidth=4, alpha=0.8)
    ax.plot(interval_2, [int2_height, int2_height], linewidth=4, alpha=0.8)
    return ax
