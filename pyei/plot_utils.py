"""Plotting functions for visualizing ei outputs"""
import warnings
import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import ticker as mticker
import numpy as np
import scipy.stats as st

__all__ = [
    "plot_precincts",
    "plot_boxplot",
    "plot_kdes",
    "plot_kde",
    "plot_conf_or_credible_interval",
]


def plot_single_ridgeplot(ax, group1_pref, group2_pref, z_init, trans, overlap=1.3, num_points=500):
    """Helper function for plot_precincts that plots a single ridgeplot (e.g.,
    for a single precinct for a given candidate.)
    Arguments:
    ax          :   matplotlib axis object
    group1_pref :   The estimates for the support for the candidate among
                    Group 1
    group2_pref :   The estimates for the support for the candidate among
                    Group 2
    z_init      :   The initial value for the z-order (helps determine
                    how plots get drawn over one another)
    trans       :   The y-translation for this plot
    Optional arguments:
    overlap     :   how much this ridgeplot may overlap with the ridgeplot
                    above it
    num_points  :   The number of evenly spaced points in [0, 1] that we
                    use to plot compute the KDE curve
    """
    x = np.linspace(0, 1, num_points)  # 500 points between 0 and 1 on the x-axis
    group1_kde = st.gaussian_kde(group1_pref)
    group2_kde = st.gaussian_kde(group2_pref)

    group1_y = group1_kde(x)
    group1_y = overlap * group1_y / group1_y.max()
    group2_y = group2_kde(x)
    group2_y = overlap * group2_y / group2_y.max()

    ax.fill_between(
        x,
        group1_y + trans,
        trans,
        color="steelblue",
        zorder=z_init,
    )
    ax.plot(x, group1_y + trans, color="black", linewidth=1, zorder=z_init + 1)

    ax.fill_between(
        x,
        group2_y + trans,
        trans,
        color="orange",
        zorder=z_init + 2,
    )
    ax.plot(x, group2_y + trans, color="black", linewidth=1, zorder=z_init + 3)


def plot_precincts(
    voting_prefs_group1,
    voting_prefs_group2,
    precinct_labels=None,
    show_all_precincts=False,
    ax=None,
):
    """Ridgeplots of sampled voting preferences for each precinct
    Arguments:
    voting_prefs_group1 :   A numpy array with shape (# of samples x
                            # of precincts) representing the estimates
                            of support for given candidate among group 1
                            in each precinct in each sample
    voting_prefs_group2 :   Same as voting_prefs_group2, except showing
                            support among group 2
    Optional arguments:
    precinct_labels     :   The names for each precinct
    show_all_precincts  :   By default, we only show the first 50 precincts.
                            If show_all_precincts is True, we plot the
                            ridgeplots for all precincts (i.e., one ridgeplot
                            for every column in the voting_prefs matrices)
    ax                  :   Matplotlib axis object
    """
    N = voting_prefs_group1.shape[1]
    if N > 50 and not show_all_precincts:
        warnings.warn(
            f"User attempted to plot {N} precinct-level voting preference "
            f"ridgeplots. Automatically restricting to first 50 precincts "
            f"(run with `show_all_precincts=True` to plot all precinct ridgeplots.)"
        )
        voting_prefs_group1 = voting_prefs_group1[:, :50]
        voting_prefs_group2 = voting_prefs_group2[:, :50]
        if precinct_labels is not None:
            precinct_labels = precinct_labels[:50]
        N = 50
    if precinct_labels is None:
        precinct_labels = range(1, N + 1)
    if ax is None:
        # adapt height of plot to the number of precincts
        _, ax = plt.subplots(figsize=(6.4, 0.2 * N))

    iterator = zip(voting_prefs_group1.T, voting_prefs_group2.T)

    for idx, (group1, group2) in enumerate(iterator, 0):
        ax.plot([0], [idx])
        trans = ax.convert_yunits(idx)
        plot_single_ridgeplot(ax, group1, group2, 4 * N - 4 * idx, trans)

    def replace_ticks_with_precinct_labels(value, pos):
        # pylint: disable=unused-argument
        # matplotlib axis tick formatter function
        idx = int(value)
        if idx < len(precinct_labels):
            return precinct_labels[idx]
        return value

    # replace y-axis ticks with precinct labels
    ax.set_yticks(np.arange(len(precinct_labels)))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(replace_ticks_with_precinct_labels))
    ax.set_title("Precinct level estimates of voting preferences")
    ax.set_xlabel("Percent vote for candidate")
    ax.set_ylabel("Precinct")
    return ax


def plot_boxplot(voting_prefs_group1, voting_prefs_group2, group1_name, group2_name, ax=None):
    """
    Horizontal boxplot of 2 groups of samples between 0 and 1
    """
    if ax is None:
        ax = plt.gca()
    samples_df = pd.DataFrame({group1_name: voting_prefs_group1, group2_name: voting_prefs_group2})
    ax = sns.boxplot(data=samples_df, orient="h", whis=[2.5, 97.5], ax=ax)
    ax.set_xlim((0, 1))
    return ax


def plot_boxplots(sampled_voting_prefs, group_names, candidate_names):
    """
    Horizontal boxplots for r x c sets of samples between 0 and 1

    sampled_voting_prefs: num_samples x r x c

    c subplots, each showing the sampled voting preference of each of r groups.
    """
    # TODO add ax argument
    _, num_groups, num_candidates = sampled_voting_prefs.shape
    fig, axes = plt.subplots(num_candidates)

    for candidate_idx in range(num_candidates):
        samples_df = pd.DataFrame(
            {group_names[i]: sampled_voting_prefs[:, i, candidate_idx] for i in range(num_groups)}
        )

        ax = axes[candidate_idx]
        sns.despine(ax=ax, left=True)
        sns.boxplot(data=samples_df, orient="h", whis=[2.5, 97.5], ax=ax)
        ax.set_xlim((0, 1))
        ax.set_title(candidate_names[candidate_idx])
        ax.tick_params(axis="y", left=False)  # remove y axis ticks

    fig.subplots_adjust(hspace=0.75)
    return ax


def plot_summary(
    voting_prefs_group1, voting_prefs_group2, group1_name, group2_name, candidate_name
):
    """ Plot KDE, histogram, and boxplot"""
    _, (ax_box, ax_hist) = plt.subplots(
        2, sharex=True, figsize=(12, 6.4), gridspec_kw={"height_ratios": (0.15, 0.85)}
    )
    sns.despine(ax=ax_hist)
    sns.despine(ax=ax_box, left=True)
    # plot custom boxplot, with two boxplots in the same row
    colors = sns.color_palette()  # fetch seaborn default color palette
    plot_props = dict(fliersize=5, linewidth=2, whis=[2.5, 97.5])
    flier1_props = dict(marker="o", markerfacecolor=colors[0], alpha=0.5)
    flier2_props = dict(marker="d", markerfacecolor=colors[1], alpha=0.5)
    sns.boxplot(
        x=voting_prefs_group1,
        orient="h",
        color=colors[0],
        ax=ax_box,
        flierprops=flier1_props,
        **plot_props,
    )
    sns.boxplot(
        x=voting_prefs_group2,
        orient="h",
        color=colors[1],
        ax=ax_box,
        flierprops=flier2_props,
        **plot_props,
    )
    ax_box.tick_params(axis="y", left=False)  # remove y axis ticks

    # plot distribution
    plot_kde(voting_prefs_group1, voting_prefs_group2, group1_name, group2_name, ax=ax_hist)
    ax_hist.set_xlabel(f"Support for {candidate_name}")
    return (ax_box, ax_hist)


def plot_kde(voting_prefs_group1, voting_prefs_group2, group1_name, group2_name, ax=None):
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


def plot_kdes(sampled_voting_prefs, group_names, candidate_names, plot_by="candidate"):
    """
    Plot a kernel density plot for prefs of voting groups for each candidate

    by: {"candidate", "group"}. If candidate, one plot per candidate, with each plot
    showing the kernel density estimates of voting preferences of all groups. If
    "group", one plot per group, with each plot showing the kernel density estimates
    of voting preferences for all candidates.

    """
    # TODO pass axes as argument
    # TODO plot by group
    _, num_groups, num_candidates = sampled_voting_prefs.shape
    if plot_by == "candidate":
        num_plots = num_candidates
        num_kdes_per_plot = num_groups
        titles = candidate_names
        legend = group_names
    elif plot_by == "group":
        num_plots = num_groups
        num_kdes_per_plot = num_candidates
        titles = group_names
        sampled_voting_prefs = np.swapaxes(sampled_voting_prefs, 1, 2)  # TODO: Check this
        legend = candidate_names
    else:
        raise ValueError("plot_by must be 'group' or 'candidate' (default: 'candidate')")
    fig, axes = plt.subplots(num_candidates, sharex=True)
    fig.subplots_adjust(hspace=0.5)
    for plot_idx in range(num_plots):
        ax = axes[plot_idx]
        for kde_idx in range(num_kdes_per_plot):
            sns.distplot(
                sampled_voting_prefs[:, kde_idx, plot_idx],
                hist=True,
                ax=ax,
                label=legend[kde_idx],
            )
        ax.set_title(titles[plot_idx])
    axes[0].legend(bbox_to_anchor=(1, 1), loc="upper left")


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


def tomography_plot(group_fraction, votes_fraction):
    # TODO: pass ax as argument
    num_precincts = len(group_fraction)
    b_1 = np.linspace(0, 1, 200)
    _, ax = plt.subplots()
    ax.set_xlim((0, 1))
    ax.set_ylim((0, 1))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("voter pref of group 1")
    ax.set_ylabel("voter pref of group 2")
    for n in range(num_precincts):
        b_2 = (votes_fraction[n] - b_1 * group_fraction[n]) / (1 - group_fraction[n])
        ax.plot(b_1, b_2, c="b")
    return ax
