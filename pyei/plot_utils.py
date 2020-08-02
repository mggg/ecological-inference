"""Plotting functions for visualizing ei outputs"""
import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import scipy.stats as st

__all__ = ["plot_precincts", "plot_boxplot", "plot_kdes", "plot_conf_or_credible_interval"]


def plot_precincts(voting_prefs_group1, voting_prefs_group2, y_labels=None, ax=None):
    """Ridgeplots of sampled voting preferences for each precinct"""
    n_x_pts = 500
    overlap = 1.3
    if ax is None:
        _, ax = plt.subplots()
    x = np.linspace(0, 1, n_x_pts)

    N = voting_prefs_group1.shape[1]
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


def plot_kdes(voting_prefs_group1, voting_prefs_group2, group1_name, group2_name, ax=None):
    """'
    Plot kernel density plots of samples between 0 and 1 (e.g. of voting preferences) for two groups
    """
    if ax is None:
        ax = plt.gca()
    ax.set_xlim((0, 1))
    sns.distplot(voting_prefs_group1, hist=True, ax=ax)
    sns.distplot(voting_prefs_group2, hist=True, ax=ax)
    text_pos_y = ax.get_ylim()[1] * 0.9
    ax.text(voting_prefs_group1.mean(), text_pos_y, group1_name)
    ax.text(voting_prefs_group2.mean(), text_pos_y, group2_name)
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
