"""Plotting functions for visualizing ei outputs"""
import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt


def plot_boxplot(
    voting_prefs_group1, voting_prefs_group2, group1_name, group2_name, ax=None
):
    '''
    Horizontal boxplot of 2 groups of samples between 0 and 1
    '''
    if ax is None:
        ax = plt.gca()
    samples_df = pd.DataFrame(
        {group1_name: voting_prefs_group1, group2_name: voting_prefs_group2}
    )
    ax = sns.boxplot(data=samples_df, orient="h", ax=ax)
    ax.set_xlim((0, 1))
    return ax


def plot_kdes(
    voting_prefs_group1, voting_prefs_group2, group1_name, group2_name, ax=None
):
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
