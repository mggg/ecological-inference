"""
Plotting functions for visualizing ei outputs

"""

import warnings
import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import ticker as mticker
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches
import numpy as np
import scipy.stats as st

__all__ = [
    "plot_boxplots",
    "plot_conf_or_credible_interval",
    "plot_intervals_all_precincts",
    "plot_kdes",
    "plot_polarization_kde",
    "plot_precinct_scatterplot",
    "plot_precincts",
    "plot_summary",
    "tomography_plot",
]

PALETTE = "Dark2"  # set library-wide color palette
FONTSIZE = 20
TITLESIZE = 24
TICKSIZE = 15
FIGSIZE = (10, 5)
colors = sns.color_palette(PALETTE)


def size_ticks(ax, axis="x"):
    """
    Helper function to size the x- or ytick (numbers) of a matplotlib Axis

    Parameters
    ----------
    axis : string
        Either 'x' or 'y', specifies which axis's ticks are being sized
    ax: matplotlib axis object
    """
    if axis == "x":
        ax.set_xlim(0, 1)
        xticks = [round(xtick, 1) for xtick in ax.get_xticks()]
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticks, size=TICKSIZE)
    elif axis == "y":
        ax.set_ylim(0, 1)
        yticks = [round(ytick, 1) for ytick in ax.get_yticks()]
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticks, size=TICKSIZE)
    else:
        raise ValueError("You need to specify an 'x' or 'y' axis!")


def size_yticklabels(ax):
    """
    Helper function to size the ytick labels of a matplotlib Axis

    Parameters
    ----------
    ax : matplotlib axis object
    """
    ax.set_yticklabels(ax.get_yticklabels(), size=TICKSIZE)


def plot_single_ridgeplot(
    ax,
    group_prefs,
    colors,  # pylint: disable=redefined-outer-name
    alpha,
    z_init,
    trans,
    overlap=1.3,
    num_points=500,
):
    """Helper function for plot_precincts that plots a single ridgeplot (e.g.,
    for a single precinct for a given candidate.)

    Parameters
    ----------
    ax : matplotlib axis object
    group_prefs: [array]
        A list where each element is an array of estimates for support
        for the candidate from a group (array of floats, expected in [0,1])
    colors : array
        The (ordered) names of colors to use to fill ridgeplots
    z_init : float
        The initial value for the z-order (helps determine
        how plots get drawn over one another)
    trans
        The y-translation for this plot
    overlap : float, optional
        how much this ridgeplot may overlap with the ridgeplot
        above it
    num_points : int, optional
        The number of evenly spaced points in [0, 1] that we
        use to plot compute the KDE curve
    """
    x = np.linspace(0, 1, num_points)  # 500 points between 0 and 1 on the x-axis
    group_kdes = [st.gaussian_kde(group_pref) for group_pref in group_prefs]

    group_ys = []
    for i in range(len(group_prefs)):
        group_y = group_kdes[i](x)
        group_y = overlap * group_y / group_y.max()
        group_ys.append(group_y)

    for i, group_y in enumerate(group_ys):
        ax.fill_between(
            x,
            group_y + trans,
            trans,
            color=colors[i],
            alpha=alpha,
            zorder=z_init,
        )
        ax.plot(
            x,
            group_y + trans,
            color="black",
            linewidth=1,
            zorder=z_init + 1 + (2 * i),
        )


def plot_single_histogram(
    ax,
    group_prefs,
    colors,  # pylint: disable=redefined-outer-name
    alpha,
    z_init,
    trans,  # pylint: disable=redefined-outer-name
):
    """Helper function for plot_precincts that plots a single precinct histogram(s)
       (i.e.,for a single precinct for a given candidate.)

    Parameters
    ----------
    ax : matplotlib axis object
    group_prefs: [array]
        A list where each element is an array of estimates for support
        for the candidate from a group (array of floats, expected in [0,1])
    colors : array
        The (ordered) names of colors to use to fill ridgeplots
    z_init : float
        The initial value for the z-order (helps determine
        how plots get drawn over one another)
    trans
        The y-translation for this plot
    """

    bins = np.linspace(0, 1.0, num=20)
    for i, group_pref in enumerate(group_prefs):
        weights, bins = np.histogram(group_pref, bins=bins)
        weights = weights / weights.max()
        ax.hist(
            bins[:-1],
            bins=bins,
            weights=weights,
            bottom=trans,
            zorder=z_init + 1,
            color=colors[i],
            alpha=alpha,
            edgecolor="black",
        )


def plot_precincts(
    voting_prefs,
    group_names,
    candidate,
    alpha=1,
    precinct_labels=None,
    show_all_precincts=False,
    plot_as_histograms=False,
    ax=None,
):
    """Ridgeplots of sampled voting preferences for each precinct

    Parameters
    ----------
    voting_prefs : list of numpy arrays
        Each element has shape (# of samples x # of precincts) representing
        the samples of support for the given candidate among a given group
        in each precinct. Each element refers to a different group.
    group_names: list of str
        The demographic group names, for display in the legend
    candidate: str
        The candidate name
    alpha: float
        The opacity for the fill color in the kdes/histograms
    precinct_labels : list of str (optional)
        The names for each precinct
    show_all_precincts : bool, optional
        By default (show_all_precincts=False), we only show the first 50
        precincts. If show_all_precincts is True, we plot the ridgeplots
        for all precincts (i.e., one ridgeplot for every column in the
        voting_prefs matrices)
    plot_as_histograms : bool, optional
        Default: False If true, plot with histograms instead of kdes
    ax : Matplotlib axis object or None, optional
        Default=None

    Returns
    -------
    ax: Matplotlib axis object
    """
    N = voting_prefs[0].shape[1]
    if N > 50 and not show_all_precincts:
        warnings.warn(
            f"User attempted to plot {N} precinct-level voting preference "
            f"ridgeplots. Automatically restricting to first 50 precincts "
            f"(run with `show_all_precincts=True` to plot all precinct ridgeplots.)"
        )
        voting_prefs = [prefs[:, :50] for prefs in voting_prefs]
        if precinct_labels is not None:
            precinct_labels = precinct_labels[:50]
        N = 50
    if precinct_labels is None:
        precinct_labels = range(1, N + 1)

    legend_space = len(group_names) + 2
    if ax is None:
        # adapt height of plot to the number of precincts
        _, ax = plt.subplots(figsize=(FIGSIZE[0], 0.3 * (N + legend_space)))

    transposed_voting_prefs = [prefs.T for prefs in voting_prefs]
    iterator = zip(*transposed_voting_prefs)

    for idx, group_prefs in enumerate(iterator, 0):
        ax.plot([0], [idx])
        trans = ax.convert_yunits(idx)
        if plot_as_histograms:
            plot_single_histogram(ax, group_prefs, colors, alpha, 4 * (N - idx), trans)
        else:
            plot_single_ridgeplot(ax, group_prefs, colors, alpha, 4 * (N - idx), trans)
    for i in range(legend_space):
        # add `legend_space` number of lines to the top of the plot for legend
        ax.plot([0], [N + i])

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
    ax.set_title("Precinct level estimates of voting preferences", fontsize=TITLESIZE)
    ax.set_xlabel(f"Percent vote for {candidate}", fontsize=FONTSIZE)
    ax.set_ylabel("Precinct", fontsize=FONTSIZE)

    proxy_handles = [
        mpatches.Patch(color=colors[i], alpha=alpha, ec="black", label=group_names[i])
        for i in range(len(group_names))
    ]
    ax.legend(handles=proxy_handles, prop={"size": 14}, loc="upper center")
    ax.set_ylim(-1, ax.get_ylim()[1])
    size_ticks(ax, "x")
    return ax


def plot_boxplots(
    sampled_voting_prefs, group_names, candidate_names, plot_by="candidate", axes=None
):
    """
    Horizontal boxplots for r x c sets of samples between 0 and 1

    Parameters
    ----------
    sampled_voting_prefs : Numpy array
        Shape is: num_samples x r x c(where r=# of demographic groups,
        c=# of voting outcomes), representing the samples of the support
        for each candidate (or voting outcome) among each group (aggregated
        across all precincts)
    group_names : list of str
        Length = r (where r=# of demographic groups), the names of the
        demographic groups (order should match order of the columns in
        sampled_voting_prefs)
    candidate_names : list of str
        Length = c (where c=# of voting outcomes), the names of the candidates
        or voting outcomes (order should match order of the last dimension of
        sampled_voting_prefs)
    plot_by : {"candidate", "group"}
        (Default='candidate')
        If 'candidate', make one plot per candidate, with each plot showing the boxplots of
        estimates of voting preferences of all groups. If 'group', one plot
        per group, with each plot showing the boxplots of estimates of voting
        preferences for all candidates.
    axes : list of Matplotlib axis object or None
        Default=None.
        If not None and plot_by is 'candidate', should have length c (number of candidates).
        If plot_by is 'group', should have length r (number of groups)

    Returns
    -------
    ax : Matplotlib axis object
        Has c subplots, each showing the sampled voting preference of each of r groups.

    Notes
    -----
    If passing existing axes within a subplot, consider using, e.g.,
    plt.subplots_adjust(hspace=0.75) to make control spacing
    """

    _, num_groups, num_candidates = sampled_voting_prefs.shape
    if plot_by == "candidate":
        num_plots = num_candidates
        num_boxes_per_plot = num_groups
        titles = candidate_names
        legend = group_names
        support = "for"
        if axes is None:
            _, axes = plt.subplots(num_candidates, figsize=FIGSIZE)

    elif plot_by == "group":
        num_plots = num_groups
        num_boxes_per_plot = num_candidates
        titles = group_names
        sampled_voting_prefs = np.swapaxes(sampled_voting_prefs, 1, 2)
        legend = candidate_names
        support = "among"
        if axes is None:
            _, axes = plt.subplots(num_groups, figsize=FIGSIZE)
    else:
        raise ValueError("plot_by must be 'group' or 'candidate' (default: 'candidate')")
    plt.gcf().subplots_adjust(hspace=1)

    for plot_idx in range(num_plots):
        samples_df = pd.DataFrame(
            {legend[i]: sampled_voting_prefs[:, i, plot_idx] for i in range(num_boxes_per_plot)}
        )
        if num_plots > 1:
            ax = axes[plot_idx]
        else:
            ax = axes
        sns.boxplot(data=samples_df, orient="h", whis=[2.5, 97.5], ax=ax, palette=colors)
        ax.set_title(f"Support {support} {titles[plot_idx]}", fontsize=TITLESIZE)
        ax.tick_params(axis="y", left=False)  # remove y axis ticks
        size_ticks(ax, "x")
        size_yticklabels(ax)

    return ax


def plot_summary(
    sampled_voting_prefs,
    group1_name,
    group2_name,
    candidate_name,
    axes=None,
):
    """Plot KDE, histogram, and boxplot for 2x2 case

    Parameters
    ----------
    sampled_voting_prefs : array
        num_samples x 2 x 1 Samples of estimated voting preferences (support)
        of group 1 for the candidate.
    group1_name : str
        Name of group 1 (used for legend of plot)
    group2_name : str
        Name of group 2 (used for legend of plot)
    candidate_name: str
        The name of the candidate
    axes : list or tuple of matplotlib axis objects or None
        Default=None
        Length 2: (ax_box, ax_hist)

    Returns
    -------
    ax_box : Matplotlib axis object
    ax_hist : Matplotlib axis object
    """
    if axes is None:
        _, (ax_hist, ax_box) = plt.subplots(
            2,
            sharex=True,
            figsize=FIGSIZE,
            gridspec_kw={"height_ratios": (0.85, 0.15)},
        )
    else:
        ax_hist, ax_box = axes
    size_ticks(ax_box, "x")
    ax_box.set_title("")
    ax_box.set_xlabel(f"Support for {candidate_name}", fontsize=FONTSIZE)
    sns.despine(ax=ax_hist)
    sns.despine(ax=ax_box, left=True)
    # plot custom boxplot, with two boxplots in the same row
    plot_props = {"fliersize": 5, "linewidth": 2, "whis": [2.5, 97.5]}
    flier1_props = {"marker": "o", "markerfacecolor": colors[0], "alpha": 0.5}
    flier2_props = {"marker": "d", "markerfacecolor": colors[1], "alpha": 0.5}
    sns.boxplot(
        x=sampled_voting_prefs[:, 0, 0],
        orient="h",
        color=colors[0],
        ax=ax_box,
        flierprops=flier1_props,
        **plot_props,
    )
    sns.boxplot(
        x=sampled_voting_prefs[:, 1, 0],
        orient="h",
        color=colors[1],
        ax=ax_box,
        flierprops=flier2_props,
        **plot_props,
    )
    ax_box.tick_params(axis="y", left=False)  # remove y axis ticks

    # plot distribution
    plot_kdes(sampled_voting_prefs, [group1_name, group2_name], [candidate_name], axes=ax_hist)
    ax_hist.set_title("EI Summary", fontsize=TITLESIZE)
    plt.subplots_adjust(hspace=0.005)
    return (ax_box, ax_hist)


def plot_precinct_scatterplot(ei_runs, run_names, candidate, demographic_group="all", ax=None):
    """
    Given two RxC EI runs, plot precinct-by-precinct comparison of preferences
    for a given candidate from a given demographic group.

    Parameters
    ----------
    ei_runs: array
        Length = 2
        Element Type = RowByColumnEI
    run_names: array
        Length = 2
        Element Type = string
        Name of each EI run (in the same order as ei_runs!)
    candidate: string
        Must be a candidate common to both EI runs
    demographic_group: string
        Must be a demographic group common to both EI runs, or "all",
        which plots and labels each demographic group onto the same axes.

    Returns
    -------
    ax: Matplotlib axis object
    """
    if ax is None:
        _, ax = plt.subplots(1, figsize=(2 * FIGSIZE[0], 2 * FIGSIZE[1]))
    ax.set_xlim((0, 1))
    ax.set_ylim((0, 1))

    prec_means1, _ = ei_runs[0].precinct_level_estimates()
    prec_means2, _ = ei_runs[1].precinct_level_estimates()

    # Set group names and candidates in case runs are TwoByTwoEI
    if not hasattr(ei_runs[0], "demographic_group_names"):  # then is TwoByTwoEI
        demographic_group_names1 = list(ei_runs[0].group_names_for_display())
        candidate_names1 = [
            ei_runs[0].candidate_name,
            "not " + ei_runs[0].candidate_name,
        ]
    else:
        demographic_group_names1 = ei_runs[0].demographic_group_names
        candidate_names1 = ei_runs[0].candidate_names
    if not hasattr(ei_runs[1], "demographic_group_names"):  # then it is TwoByTwoEI
        demographic_group_names2 = list(ei_runs[1].group_names_for_display())
        candidate_names2 = [
            ei_runs[1].candidate_name,
            "not " + ei_runs[1].candidate_name,
        ]
    else:
        demographic_group_names2 = ei_runs[1].demographic_group_names
        candidate_names2 = ei_runs[1].candidate_names

    common_groups = [g for g in demographic_group_names1 if g in demographic_group_names2]

    group_dict = {}
    for group in common_groups:
        group_dict[group] = (
            prec_means1[
                :,
                demographic_group_names1.index(group),
                candidate_names1.index(candidate),
            ],
            prec_means2[
                :,
                demographic_group_names2.index(group),
                candidate_names2.index(candidate),
            ],
        )

    if demographic_group == "all":
        for k, (x_vals, y_vals) in group_dict.items():
            sns.scatterplot(x=x_vals, y=y_vals, label=k)
    else:
        sns.scatterplot(
            x=group_dict[demographic_group][0],
            y=group_dict[demographic_group][1],
            label=demographic_group,
        )
    sns.lineplot(x=[0, 1], y=[0, 1], alpha=0.5, color="grey")
    ax.set_title(
        f"{run_names[0]} vs. {run_names[1]}\n Predicted support for {candidate}",
        fontsize=TITLESIZE,
    )
    ax.set_xlabel(f"Support for {candidate} (from {run_names[0]})", fontsize=FONTSIZE)
    ax.set_ylabel(f"Support for {candidate} (from {run_names[1]})", fontsize=FONTSIZE)
    ax.legend()
    size_ticks(ax, "x")
    size_ticks(ax, "y")

    return ax


def plot_margin_kde(group, candidates, samples, thresholds, percentile, show_threshold, ax):
    """
    Plots a kde for the margin between two candidates among a given demographic group

    Parameters:
    -----------
    samples: array
        samples of the differences in voting preferences (candidate 1 - candidate 2)
    thresholds: array
        a list of thresholds for the difference in voting patterns between two groups
    group: str
        the name of the demographic group in question
    candidates : list of str
        the names of the two candidates in question
    show_threshold: bool
        if true, add vertical lines at the threshold on the plot

    Returns
    -------
    ax: Matplotlib axis object
    """
    if ax is None:
        _, ax = plt.subplots(figsize=FIGSIZE)

    sns.histplot(
        samples,
        kde=True,
        ax=ax,
        element="step",
        stat="density",
        color="steelblue",
        linewidth=0,
    )
    ax.set_ylabel("Density", fontsize=FONTSIZE)
    if len(thresholds) == 1:
        threshold_string = f"> {thresholds[0]:.2f}"
    else:
        threshold_string = f"in [{thresholds[0]:.2f}, {thresholds[1]:.2f}]"
    if show_threshold:
        for threshold in thresholds:
            ax.axvline(threshold, c="gray")
        if len(thresholds) == 2:
            ax.axvspan(thresholds[0], thresholds[1], facecolor="gray", alpha=0.2)
        else:
            ax.axvspan(thresholds[0], 1, facecolor="gray", alpha=0.2)
        ax.text(
            thresholds[-1] + 0.05,
            0.5,
            f"Prob (margin {threshold_string} ) = {percentile:.1f}%",
            fontsize=FONTSIZE,
        )

    ax.set_title(f"{candidates[0]} - {candidates[1]} margin among {group}", fontsize=TITLESIZE)
    ax.set_xlabel(f"{group} support for {candidates[0]} - {candidates[1]}", fontsize=FONTSIZE)
    ax.set_xlim((-1, 1))
    xticks = ax.get_xticks()
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks, size=TICKSIZE)


def plot_polarization_kde(
    diff_samples,
    thresholds,
    probability,
    groups,
    candidate_name,
    show_threshold=False,
    ax=None,
    color="steelblue",
):
    """
    Plots a kde for the differences in voting preferences between two groups

    Parameters:
    -----------
    diff_samples: array
        samples of the differences in voting preferences (group_complement - group)
    probability: float
        the probability that (group_complement - group) > threshold
    thresholds: array
        a list of thresholds for the difference in voting patterns between two groups
    groups: list
        the names of the two groups being compared
    candidate_name : string
        the name of the candidate or voting outcome whose support is shown in the kde
    show_threshold: bool
        if true, add a vertical line at the threshold on the plot and display the associated
        tail probability
    color: str
        specifies a color for matplotlib to be used in the histogram/kde

    Returns
    -------
    ax: Matplotlib axis object
    """

    if ax is None:
        _, ax = plt.subplots(figsize=FIGSIZE)

    sns.histplot(
        diff_samples,
        kde=True,
        ax=ax,
        element="step",
        stat="density",
        label=groups[0] + " - " + groups[1],
        color=color,
        linewidth=0,
    )
    ax.set_ylabel("Density", fontsize=FONTSIZE)
    if len(thresholds) == 1:
        threshold_string = f"> {thresholds[0]:.2f}"
    else:
        threshold_string = f"in [{thresholds[0]:.2f}, {thresholds[1]:.2f}]"
    if show_threshold:
        for threshold in thresholds:
            ax.axvline(threshold, c="gray")
        if len(thresholds) == 2:
            ax.axvspan(thresholds[0], thresholds[1], facecolor="gray", alpha=0.2)
        else:
            ax.axvspan(thresholds[0], 1, facecolor="gray", alpha=0.2)
        ax.text(
            thresholds[-1] + 0.05,
            0.5,
            f"Prob (difference {threshold_string} ) = {probability:.1f}%",
            fontsize=FONTSIZE,
        )

    ax.set_title(f"Polarization KDE for {candidate_name}", fontsize=TITLESIZE)
    ax.set_xlabel(f"({groups[0]} - {groups[1]}) support for {candidate_name}", fontsize=FONTSIZE)
    ax.set_xlim((-1, 1))
    xticks = ax.get_xticks()
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks, size=TICKSIZE)

    return ax


def plot_kdes(sampled_voting_prefs, group_names, candidate_names, plot_by="candidate", axes=None):
    """
    Plot a kernel density plot for prefs of voting groups for each candidate

    Parameters
    ----------
    sampled_voting_prefs : numpy array
        Shape: num_samples x r x c (where r = # of demographic groups, c= #
        of candidates or voting outcomes). Gives samples of support from each group
        for each candidate. NOTE: for a 2 x 2 case where we have just two candidates/outcomes
        and want only one plot, have c=1
    group_names : list of str
        Names of the demographic groups (length r)
    candidate_names : list of str
        Names of the candidates or voting outcomes (length c)
    plot_by : {"candidate", "group"}
        (Default='candidate')
        If 'candidate', make one plot per candidate, with each plot showing the kernel
        density estimates of voting preferences of all groups. If 'group', one plot
        per group, with each plot showing the kernel density estimates of voting
        preferences for all candidates.
    axes : list of Matplotlib axis object or None
        Default=None.
        If not None and plot_by is 'candidate', should have length c (number of candidates).
        If plot_by is 'group', should have length r (number of groups)

    Returns
    -------
    ax : Matplotlib axis object
    """

    _, num_groups, num_candidates = sampled_voting_prefs.shape
    if plot_by == "candidate":
        num_plots = num_candidates
        num_kdes_per_plot = num_groups
        titles = candidate_names
        legend = group_names
        support = "for"
        if axes is None:
            _, axes = plt.subplots(num_candidates, figsize=FIGSIZE, sharex=True)
            plt.subplots_adjust(hspace=0.5)
    elif plot_by == "group":
        num_plots = num_groups
        num_kdes_per_plot = num_candidates
        titles = group_names
        sampled_voting_prefs = np.swapaxes(sampled_voting_prefs, 1, 2)
        legend = candidate_names
        support = "among"
        if axes is None:
            _, axes = plt.subplots(num_groups, figsize=FIGSIZE, sharex=True)
            plt.subplots_adjust(hspace=0.5)
    else:
        raise ValueError("plot_by must be 'group' or 'candidate' (default: 'candidate')")

    middle_plot = int(np.floor(num_plots / 2))
    for plot_idx in range(num_plots):
        if num_plots > 1:
            ax = axes[plot_idx]
            axes[middle_plot].set_ylabel("Probability Density", fontsize=FONTSIZE)
        else:
            ax = axes
            axes.set_ylabel("Probability Density", fontsize=FONTSIZE)
        ax.set_title(f"Support {support} " + titles[plot_idx], fontsize=TITLESIZE)
        ax.set_xlim((0, 1))
        size_ticks(ax, "x")

        for kde_idx in range(num_kdes_per_plot):
            sns.histplot(
                sampled_voting_prefs[:, kde_idx, plot_idx],
                kde=True,
                ax=ax,
                stat="density",
                element="step",
                label=legend[kde_idx],
                color=colors[kde_idx],
                linewidth=0,
            )
            ax.set_ylabel("")

    if num_plots > 1:
        axes[middle_plot].legend(bbox_to_anchor=(1, 1), loc="upper left", prop={"size": 12})
    else:
        ax.legend(prop={"size": 12})
    return axes


def plot_conf_or_credible_interval(intervals, group_names, candidate_name, title, ax=None):
    """
    Plot confidence of credible interval for two different groups

    Parameters
    ----------
    intervals : list of arrays or tuple
        Length is number of demographic groups, each element in the array
        is a length-two array that gives
        (lower, upper) bounds for credible or confidence interval
        for support from group 1 for the candidate of interest
    groups_names : list of str
        Names of groups (ordered to match order of intervals), for plot legend
    candidate_name : str
        Name of candidate (or voting outcome) whose support is to be plotted
    title : str
        Title of plot
    ax : Matplotlib axis object or None, optional
        Default=None

    Returns
    -------
    ax : Matplotlib axis object
    """

    if ax is None:
        _, ax = plt.subplots(figsize=FIGSIZE)

    int_heights = 0.2 * np.arange(len(group_names), 0, -1)
    ax.set_ylim(0, (len(group_names) + 1) * 0.2)
    ax.set_title(title, fontsize=TITLESIZE)
    ax.set_xlabel(f"Support for {candidate_name}", fontsize=FONTSIZE)
    ax.set(
        xlim=(0, 1),
        frame_on=True,
        aspect=0.3,
    )

    ax.get_xaxis().tick_bottom()
    ax.axes.get_yaxis().set_visible(False)
    ax.grid()
    for idx, group_name in enumerate(group_names):
        ax.text(1, int_heights[idx], f" {group_name}", fontsize=FONTSIZE)
        ax.plot(
            intervals[idx],
            [int_heights[idx], int_heights[idx]],
            linewidth=10,
            alpha=0.8,
            color=colors[idx],
        )
    size_ticks(ax, "x")

    return ax


def plot_intervals_all_precincts(
    point_estimates,
    intervals,
    candidate_name,
    precinct_labels,
    title,
    ax=None,
    show_all_precincts=False,
):
    """
    Plot intervals&point estimates of support for candidate, sorted by point estimates for precincts

    Parameters
    ----------
    point_estimates: array
        Array of length num_precincts, each element is the estimate of support for the
        candidate among the demographic group of interest in that precinct
    intervals: array
        Array of arrays (shape num_precincts, each elt an array of length 2)
        Each element is an array of length two giving the lower, upper limits of a credible
        or confidence interval for support for the candidate in the precinct
    candidate_name: str
        Name of candidate, used in x axis label
    title: str
        Title for plot
    ax : Matplotlib axis object or None, optional
        Default=None
    show_all_precincts: bool, optional
        (default=False). If True, show estimates&intervals for all precincts. If False,
        show only the first 50, for readibility.

    Returns
    -------
    ax : Matplotlib axis object
    """
    num_intervals = len(point_estimates)

    if num_intervals > 50 and not show_all_precincts:
        warnings.warn(
            f"User attempted to plot {num_intervals} precinct-level voting preference "
            f"ridgeplots. Automatically restricting to first 50 precincts "
            f"(run with `show_all_precincts=True` to plot all precinct ridgeplots.)"
        )
        point_estimates = point_estimates[:50]
        intervals = intervals[:, :50]
        if precinct_labels is not None:
            precinct_labels = precinct_labels[:50]
        num_intervals = 50

    int_heights = 20 * np.arange(num_intervals) + 20
    tot_height = int_heights[-1] + 20
    int_heights = tot_height - int_heights

    if ax is None:
        _, ax = plt.subplots(
            1, frameon=False, constrained_layout=True, figsize=(16, num_intervals / 4)
        )

    if precinct_labels is None:
        precinct_labels = range(num_intervals)

    ax.set(
        title=title,
        xlim=(0, 1),
        ylim=(0, tot_height),
        xlabel=f"Support for {candidate_name}",
        frame_on=False,
    )

    ax.get_xaxis().tick_bottom()
    ax.axes.get_yaxis().set_visible(False)

    point_estimates, intervals, precinct_labels = zip(
        *sorted(zip(point_estimates, intervals, precinct_labels))
    )
    bars = []

    for point_estimate, interval, precinct_label, int_height in zip(
        point_estimates, intervals, precinct_labels, int_heights
    ):
        width = interval[1] - interval[0]
        height = 16
        lower_left = (interval[0], int_height)
        bars.append(Rectangle(lower_left, width, height))

        ax.scatter(point_estimate, int_height + height / 2, s=30, alpha=1, c="k")
        ax.text(1, int_height + height / 2, precinct_label, fontsize=12)

    pc_bars = PatchCollection(bars, facecolor="gray", alpha=0.6)
    ax.add_collection(pc_bars)

    return ax


def tomography_plot(
    group_fraction,
    votes_fraction,
    demographic_group_name,
    candidate_name,
    ax=None,
    c="b",
    **plot_kwargs,
):
    """Tomography plot (basic), applicable for 2x2 ei

    Parameters
    ----------
    group_fraction : array
        Array of length num_precincts, giving fraction of population of interest
        in each precinct represented by demographic group of interest
    votes_fraction: array
        An array of length num_precincts giving fraction  of votes in each
        precinct for candidate of interest
    demographic_group_name : str
        Name of demographic group of interest
    candidate_name : str
        Name of candidate or voting outcome of interest
    ax : Matplotlib axis object or None, optional
        Default=None
    c : specifies a color for Matplotlib, optional
        Default="b"
    **plot_kwargs
        Additional keyword arguments to be passed to matplotlib.Axes.plot()

    Returns
    -------
    ax : Matplotlib axis object
    """

    if ax is None:
        _, ax = plt.subplots()

    num_precincts = len(group_fraction)
    b_1 = np.linspace(0, 1, 200)
    ax.set_xlim((0, 1))
    ax.set_ylim((0, 1))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(f"voter pref of {demographic_group_name} for {candidate_name}")
    ax.set_ylabel(f"voter pref of non-{demographic_group_name} for {candidate_name}")
    for i in range(num_precincts):
        b_2 = (votes_fraction[i] - b_1 * group_fraction[i]) / (1 - group_fraction[i])
        ax.plot(b_1, b_2, c=c, **plot_kwargs)
    return ax
