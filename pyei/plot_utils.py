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
import numpy as np
import scipy.stats as st

__all__ = [
    "plot_boxplot",
    "plot_boxplots",
    "plot_conf_or_credible_interval",
    "plot_intervals_all_precincts",
    "plot_kdes",
    "plot_kde",
    "plot_precincts",
    "plot_summary",
    "tomography_plot",
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

    Parameters
    ----------
    voting_prefs_group1 : numpy array
        Shape (# of samples x # of precincts) representing the samples
        of support for given candidate among group 1 in each precinct
    voting_prefs_group2 : numpy array
        Same as voting_prefs_group2, except showing support among group 2
    precinct_labels : list of str (optional)
        The names for each precinct
    show_all_precincts : bool, optional
        By default (show_all_precincts=False), we only show the first 50 
        precincts. If show_all_precincts is True, we plot the ridgeplots
        for all precincts (i.e., one ridgeplot for every column in the
        voting_prefs matrices)
    ax : Matplotlib axis object or None, optional
        Default=None

    Returns
    -------
    ax: Matplotlib axis object
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

    Parameters
    ----------
    voting_prefs_group1 : numpy array
        shape (# of samples x # of precincts) representing the estimates of support for 
        given candidate among group 1 in each precinct in each sample
    voting_prefs_group2 : numpy array
        Same as voting_prefs_group2, except showing support among group 2
    group1_name : str
        Name of group 1 (used for legend of plot)
    group2_name : str
        Name of group 2 (used for legend of plot)
    ax : Matplotlib axis object or None, optional
        Default=None

    Returns
    -------
    Matplotlib axis object
    """
    if ax is None:
        ax = plt.gca()
    samples_df = pd.DataFrame({group1_name: voting_prefs_group1, group2_name: voting_prefs_group2})
    ax = sns.boxplot(data=samples_df, orient="h", whis=[2.5, 97.5], ax=ax)
    ax.set_xlim((0, 1))
    return ax


def plot_boxplots(sampled_voting_prefs, group_names, candidate_names, plot_by='candidate', axes=None):
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
        if axes is None:
            fig, axes = plt.subplots(num_candidates)
            
    elif plot_by == "group":
        num_plots = num_groups
        num_boxes_per_plot = num_candidates
        titles = group_names
        sampled_voting_prefs = np.swapaxes(sampled_voting_prefs, 1, 2)
        legend = candidate_names
        if axes is None:
            fig, axes = plt.subplots(num_groups)
    else:
        raise ValueError("plot_by must be 'group' or 'candidate' (default: 'candidate')")
    fig.subplots_adjust(hspace=0.75)

    for plot_idx in range(num_plots):
        samples_df = pd.DataFrame(
            {legend[i]: sampled_voting_prefs[:, i, plot_idx] for i in range(num_boxes_per_plot)}
        )
        ax = axes[plot_idx]
        sns.despine(ax=ax, left=True)
        sns.boxplot(data=samples_df, orient="h", whis=[2.5, 97.5], ax=ax)
        ax.set_xlim((0, 1))
        ax.set_title(titles[plot_idx])
        ax.tick_params(axis="y", left=False)  # remove y axis ticks

    return ax


def plot_summary(
    voting_prefs_group1, voting_prefs_group2, group1_name, group2_name, candidate_name, axes=None
):
    """ Plot KDE, histogram, and boxplot for 2x2 case

    Parameters
    ----------
    voting_prefs_group1 : array
        Length=num_samples. Samples of estimated voting preferences (support)
        of group 1 for the candidate.
    voting_prefs_group2 : array
        Length=num_samples. Same as voting_prefs_group1 but for group 2
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
        _, (ax_box, ax_hist) = plt.subplots(
            2, sharex=True, figsize=(12, 6.4), gridspec_kw={"height_ratios": (0.15, 0.85)}
        )
    else:
        ax_box, ax_hist = axes
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

    Parameters
    ----------
    voting_prefs_group1 : array
        Length=num_samples. Samples of estimated voting preferences (support)
        of group 1 for the candidate.
    voting_prefs_group2 : array
        Length=num_samples. Same as voting_prefs_group1 but for group 2
    group1_name : str
        Name of group 1 (used for legend of plot)
    group2_name : str
        Name of group 2 (used for legend of plot)
    ax : Matplotlib axis object or None
        Default=None

    Returns
    -------
    ax : Matplotlib axis object
    """
    if ax is None:
        ax = plt.gca()
    ax.set_xlim((0, 1))

    sns.histplot(
        voting_prefs_group1,
        kde=True,
        ax=ax,
        element="step",
        stat="density",
        label=group1_name,
        color=f"C{0}",
        linewidth=0,
    )
    sns.histplot(
        voting_prefs_group2,
        kde=True,
        ax=ax,
        element="step",
        stat="density",
        label=group2_name,
        color=f"C{1}",
        linewidth=0,
    )

    ax.legend()
    return ax


def plot_kdes(sampled_voting_prefs, group_names, candidate_names, plot_by="candidate", axes=None):
    """
    Plot a kernel density plot for prefs of voting groups for each candidate

    Parameters
    ----------
    sampled_voting_prefs : numpy array
        Shape: num_samples x r x c (where r = # of demographic groups, c= #
        of candidates or voting outcomes). Gives samples of support from each group
        for each candidate
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
        if axes is None:
            fig, axes = plt.subplots(num_candidates, sharex=True)
    elif plot_by == "group":
        num_plots = num_groups
        num_kdes_per_plot = num_candidates
        titles = group_names
        sampled_voting_prefs = np.swapaxes(sampled_voting_prefs, 1, 2)
        legend = candidate_names
        if axes is None:
            fig, axes = plt.subplots(num_groups, sharex=True)
    else:
        raise ValueError("plot_by must be 'group' or 'candidate' (default: 'candidate')")
    fig.subplots_adjust(hspace=0.5)
    for plot_idx in range(num_plots):
        ax = axes[plot_idx]
        for kde_idx in range(num_kdes_per_plot):
            sns.histplot(
                sampled_voting_prefs[:, kde_idx, plot_idx],
                kde=True,
                ax=ax,
                stat="density",
                element="step",
                label=legend[kde_idx],
                color=f"C{kde_idx}",
                linewidth=0,
            )

        ax.set_title(titles[plot_idx])
    axes[0].legend(bbox_to_anchor=(1, 1), loc="upper left")
    return ax


def plot_conf_or_credible_interval(
    intervals, group_names, candidate_name, title, ax=None
):
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
        ax = plt.axes(frameon=False)
    
    int_heights = 0.2 * np.arange(len(group_names), 0, -1) 

    ax.set(
        title=title,
        xlim=(0, 1),
        ylim=(0, int_heights[0] + 0.1),
        xlabel=f"Support for {candidate_name}",
        frame_on=False,
        aspect=0.3,
    )

    ax.get_xaxis().tick_bottom()
    ax.axes.get_yaxis().set_visible(False)
    for idx in range(len(group_names)):
        ax.text(1, int_heights[idx], group_names[idx])
        ax.plot(intervals[idx], [int_heights[idx], int_heights[idx]], linewidth=4, alpha=0.8)

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
    group_fraction, votes_fraction, demographic_group_name, candidate_name, ax=None
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
        ax.plot(b_1, b_2, c="b")
    return ax
