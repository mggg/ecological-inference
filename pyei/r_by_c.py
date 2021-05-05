"""
Models and fitting for rxc methods
where r and c are greater than or
equal to 2

TODO: Investigate better or reparametrized priors for multinomial-dir
TODO: Greiner-Quinn Model
TODO: Refactor to integrate with two_by_two
"""


import warnings
import pymc3 as pm
import theano.tensor as tt
import numpy as np
from .plot_utils import (
    plot_boxplots,
    plot_kdes,
    plot_intervals_all_precincts,
    plot_polarization_kde,
)

__all__ = ["ei_multinom_dirichlet_modified", "ei_multinom_dirichlet", "RowByColumnEI"]


def ei_multinom_dirichlet(group_fractions, votes_fractions, precinct_pops, lmbda1=4, lmbda2=2):
    """
    An implementation of the r x c dirichlet/multinomial EI model

    Parameters
    ----------
    group_fractions: r x num_precincts  matrix giving demographic information
        as the fraction of precinct_pop in the demographic group of interest for each of
        p precincts and r demographic groups (sometimes denoted X)
    votes_fractions: c x num_precincts matrix giving the fraction of each precinct_pop that votes
        for each of c candidates (sometimes denoted T)
    precinct_pops: Length-num_precincts vector giving size of each precinct population of interest
        (e.g. voting population) (sometimes denoted N)
    lmbda1: float parameter passed to the Gamma(lmbda, 1/lmbda2) distribution
    lmbda2: float parameter passed to the Gamma(lmbda, 1/lmbda2) distribution

    Returns
    -------
    model: A pymc3 model
    """

    num_precincts = len(precinct_pops)  # number of precincts
    num_rows = group_fractions.shape[0]  # number of demographic groups (r)
    num_cols = votes_fractions.shape[0]  # number of candidates or voting outcomes (c)

    # reshaping and rounding
    votes_count_obs = np.swapaxes(
        votes_fractions * precinct_pops, 0, 1
    ).round()  # num_precincts x r
    group_fractions_extended = np.expand_dims(group_fractions, axis=2)
    group_fractions_extended = np.repeat(group_fractions_extended, num_cols, axis=2)
    group_fractions_extended = np.swapaxes(group_fractions_extended, 0, 1)
    # num_precincts x r x c

    with pm.Model() as model:
        # TODO: are the prior conc_params what is in the literature? is it a good choice?
        # TODO: make b vs. beta naming consistent
        # conc_params = pm.Exponential("conc_params", lam=lmbda, shape=(num_rows, num_cols))
        conc_params = pm.Gamma(
            "conc_params", alpha=lmbda1, beta=1 / lmbda2, shape=(num_rows, num_cols)
        )  # chosen to match eiPack
        beta = pm.Dirichlet("b", a=conc_params, shape=(num_precincts, num_rows, num_cols))
        # num_precincts x r x c
        theta = (group_fractions_extended * beta).sum(axis=1)
        pm.Multinomial(
            "votes_count", n=precinct_pops, p=theta, observed=votes_count_obs
        )  # num_precincts x r
    return model


def ei_multinom_dirichlet_modified(
    group_fractions, votes_fractions, precinct_pops, pareto_scale=5, pareto_shape=1
):
    """
    An implementation of the r x c dirichlet/multinomial EI model with reparametrized hyperpriors

    Parameters
    ----------
    group_fractions: r x num_precincts  matrix giving demographic information
        as the fraction of precinct_pop in the demographic group of interest for each of
        p precincts and r demographic groups (sometimes denoted X)
    votes_fractions: c x num_precincts matrix giving the fraction of each precinct_pop that votes
        for each of c candidates (sometimes denoted T)
    precinct_pops: Length-num_precincts vector giving size of each precinct population of interest
        (e.g. voting population) (sometimes denoted N)

    Returns
    -------
    model: A pymc3 model

    Notes
    -----
    Reparametrizing of the hyperpriors to give (hopefully) better geometry for sampling.
    Also gives intuitive interpretation of hyperparams as mean and counts
    """

    num_precincts = len(precinct_pops)  # number of precincts
    num_rows = group_fractions.shape[0]  # number of demographic groups (r)
    num_cols = votes_fractions.shape[0]  # number of candidates or voting outcomes (c)

    # reshaping and rounding
    votes_count_obs = np.swapaxes(
        votes_fractions * precinct_pops, 0, 1
    ).round()  # num_precincts x r
    group_fractions_extended = np.expand_dims(group_fractions, axis=2)
    group_fractions_extended = np.repeat(group_fractions_extended, num_cols, axis=2)
    group_fractions_extended = np.swapaxes(group_fractions_extended, 0, 1)
    # num_precincts x r x c

    with pm.Model() as model:
        # TODO: make b vs. beta naming consistent
        kappa = pm.Pareto("kappa", alpha=pareto_shape, m=pareto_scale, shape=num_rows)  # size r
        phi = pm.Dirichlet("phi", a=np.ones(num_cols), shape=(num_rows, num_cols))  # r x c
        phi_kappa = pm.Deterministic("phi_kappa", tt.transpose(kappa * tt.transpose(phi)))
        beta = pm.Dirichlet("b", a=phi_kappa, shape=(num_precincts, num_rows, num_cols))
        # num_precincts x r x c
        theta = (group_fractions_extended * beta).sum(axis=1)  # sum across num_rows
        pm.Multinomial(
            "votes_count", n=precinct_pops, p=theta, observed=votes_count_obs
        )  # num_precincts x r
    return model


class RowByColumnEI:
    """
    Fitting and plotting for RxC models, fit via sampling
    """

    def __init__(self, model_name, **additional_model_params):
        # model_name can be 'multinomial-dirichlet' or 'greiner-quinn'
        # TODO: implement greiner quinn
        self.model_name = model_name
        self.additional_model_params = additional_model_params

        self.demographic_group_fractions = None
        self.votes_fractions = None
        self.precinct_pops = None
        self.precinct_names = None
        self.demographic_group_names = None
        self.candidate_names = None
        self.sim_model = None
        self.sim_trace = None
        self.sampled_voting_prefs = None
        self.posterior_mean_voting_prefs = None
        self.credible_interval_95_mean_voting_prefs = None
        self.num_groups_and_num_candidates = [None, None]

    def fit(
        self,
        group_fractions,
        votes_fractions,
        precinct_pops,
        demographic_group_names=None,
        candidate_names=None,
        target_accept=0.99,
        tune=1500,
        draw_samples=True,
        precinct_names=None,
        **other_sampling_args,
    ):
        """Fit the specified model using MCMC sampling
        Required arguments:
        group_fractions :   r x p (p =#precincts = num_precincts) matrix giving demographic
            information as the fraction of precinct_pop in the demographic group for each
            of p precincts and r demographic groups (sometimes denoted X)
        votes_fractions  :  c x p giving the fraction of each precinct_pop that votes
            for each of c candidates (sometimes denoted T)
        precinct_pops   :   Length-p array of ints giving size of each precinct population
                            of interest (e.g. voting population) (someteimes denoted N)
        Optional arguments:
        demographic_group_names  :  Names of the r demographic group of interest,
                                    where results are computed for the
                                    demographic group and its complement
        candidate_names          :  Name of the c candidates or voting outcomes of interest
        precinct_names          :   Length p vector giving the string names
                                    for each precinct.
        target_accept : float
            Strictly between zero and 1 (should be close to 1). Passed to pymc's
            sampling.sample
        tune : int
            Passed to pymc's sampling.sample
        draw_samples: bool, optional
            Default=True. Set to False to only set up the variable but not generate
            posterior samples (i.e. if you want to generate prior predictive samples only)
        other_sampling_args :
            For to pymc's sampling.sample
            https://docs.pymc.io/api/inference.html

        """
        # Additional params for hyperparameters
        # TODO: describe hyperparameters

        self.demographic_group_fractions = group_fractions
        self.votes_fractions = votes_fractions

        # check that precinct_pops are integers
        if not all(isinstance(p, (int, np.integer)) for p in precinct_pops):
            raise ValueError("all elements of precinct_pops must be integer-valued")
        self.precinct_pops = precinct_pops

        # give demographic groups, candidates 1-indexed numbers as names if names are not specified
        if demographic_group_names is None:
            demographic_group_names = [str(i) for i in range(1, group_fractions.shape[0] + 1)]
        if candidate_names is None:
            candidate_names = [str(i) for i in range(1, votes_fractions.shape[0] + 1)]
        self.demographic_group_names = demographic_group_names
        self.candidate_names = candidate_names

        # Set precinct names
        if precinct_names is not None:  # pylint: disable=duplicate-code
            assert len(precinct_names) == len(precinct_pops)  # pylint: disable=duplicate-code
            if len(set(precinct_names)) != len(precinct_names):  # pylint: disable=duplicate-code
                warnings.warn(
                    "Precinct names are not unique. This may interfere with "
                    "passing precinct names to precinct_level_plot()."
                )
            self.precinct_names = precinct_names

        self.num_groups_and_num_candidates = [
            group_fractions.shape[0],
            votes_fractions.shape[0],
        ]  # [r, c]

        check_dimensions_of_input(
            group_fractions,
            votes_fractions,
            precinct_pops,
            demographic_group_names,
            candidate_names,
            self.num_groups_and_num_candidates,
        )

        if self.model_name == "multinomial-dirichlet":
            self.sim_model = ei_multinom_dirichlet(
                group_fractions,
                votes_fractions,
                precinct_pops,
                **self.additional_model_params,
            )

        elif self.model_name == "multinomial-dirichlet-modified":
            self.sim_model = ei_multinom_dirichlet_modified(
                group_fractions,
                votes_fractions,
                precinct_pops,
                **self.additional_model_params,
            )
        else:
            raise ValueError(
                f"""{self.model_name} is not a supported model_name
            Currently supported: RxC models: 'multinomial-dirichlet-modified',
            'multinomial-dirichlet' """
            )

        if draw_samples:
            with self.sim_model:
                self.sim_trace = pm.sample(
                    target_accept=target_accept, tune=tune, **other_sampling_args
                )

            self.calculate_summary()

    def calculate_summary(self):
        """Calculate point estimates (post. means) and 95% equal-tailed credible intervals"""
        # multiply sample proportions by precinct pops to get samples of
        # number of voters of the demographic group who voted for the candidate
        # in each precinct
        # self.sim_trace.get_values("b") is num_samples x num_precincts x r x c
        b_reshaped = np.swapaxes(
            self.sim_trace.get_values("b"), 1, 2
        )  # num_samples x r x num_precincts x c
        b_reshaped = np.swapaxes(b_reshaped, 2, 3)  # num_samples x r x c x num_precincts
        samples_converted_to_pops = (
            b_reshaped * self.precinct_pops
        )  # num_samples x r x c num_precincts

        # obtain samples of total votes summed across all precinct for each candidate and each group
        samples_of_votes_summed_across_district = samples_converted_to_pops.sum(
            axis=3
        )  # num_samples x r x c

        # obtain samples of the districtwide proportion of each demog. group voting for candidate
        self.sampled_voting_prefs = (
            samples_of_votes_summed_across_district / self.precinct_pops.sum()
        )  # sampled voted prefs across precincts,  num_samples x r x c

        # compute point estimates
        self.posterior_mean_voting_prefs = self.sampled_voting_prefs.mean(axis=0)  # r x c

        # compute credible intervals
        percentiles = [2.5, 97.5]
        self.credible_interval_95_mean_voting_prefs = np.zeros(
            (
                self.num_groups_and_num_candidates[0],
                self.num_groups_and_num_candidates[1],
                2,
            )
        )
        for row in range(self.num_groups_and_num_candidates[0]):
            for col in range(self.num_groups_and_num_candidates[1]):
                self.credible_interval_95_mean_voting_prefs[row][col][:] = np.percentile(
                    self.sampled_voting_prefs[:, row, col], percentiles
                )

    def _calculate_polarization(self, groups, candidate, threshold=None, percentile=None):
        """
        Calculate percentile given a threshold, or vice versa.
        Exactly one of {percentile, threshold} must be None.
        Parameters:
        -----------
        groups: array-like
            Length 2 vector of demographic groups from which to calculate polarization
        candidate: string
            Candidate for which to calculate polarization
        threshold: float (optional)
            A specified level of difference in support for the candidate
            between one group and the other. If specified, use the threshold
            to calculate the percentile (exactly one of threshold and percentile
            must be None)
        percentile: float (optional)
            Between 0 and 100. Used to calculate the equal-tailed interval
            for the polarization. At least one of threshold and percentile
            must be None
        """

        candidate_index = self.candidate_names.index(candidate)
        group_index_0 = self.demographic_group_names.index(groups[0])
        group_index_1 = self.demographic_group_names.index(groups[1])

        samples = (
            self.sampled_voting_prefs[:, group_index_0, candidate_index]
            - self.sampled_voting_prefs[:, group_index_1, candidate_index]
        )

        if percentile is None and threshold is not None:
            percentile = 100 * (samples > threshold).sum() / len(self.sampled_voting_prefs)
        elif threshold is None and percentile is not None:
            threshold = np.percentile(samples, 100 - percentile)
        else:
            raise ValueError(
                """Exactly one of threshold or percentile must be None.
            Set a threshold to calculate the associated percentile, or a percentile
            to calculate the associated threshold.
            """
            )
        return threshold, percentile, samples, groups, candidate

    def polarization_report(self, groups, candidate, threshold=None, percentile=None, verbose=True):
        """
        For a given threshold, return the probability that the difference between
        the two demographic groups' preferences for the candidate is greater than
        the threshold
        OR
        For a given confidence level, calculate the associated confidence interval
        of the difference between the two groups' preferences.
        Exactly one of {percentile, threshold} must be None.
        Parameters:
        -----------
        groups:
            Length 2 vector of demographic groups from which to calculate polarization
        candidate: string
            Candidate for which to calculate polarization
        threshold: float (optional)
            A specified level of difference in support for the candidate
            between one group and the other. If specified, use the threshold
            to calculate the percentile (exactly one of threshold and percentile
            must be None)
        percentile: float (optional)
            Between 0 and 100. Used to calculate the equal-tailed interval
            for the polarization. At least one of threshold and percentile
            must be None
        verbose: bool
            If true, print a report putting polarization in context

        """
        return_interval = threshold is None

        if not all(group in self.demographic_group_names for group in groups):
            raise ValueError(
                f"""Elements of group_names must be in the list of demographic_group_names
                provided to fit():
                {self.demographic_group_names}"""
            )

        if candidate not in self.candidate_names:
            raise ValueError(
                f"""candidate_name must be in the list of candidate_names provided to fit():
                {self.candidate_names}"""
            )

        if return_interval:
            lower_percentile = (100 - percentile) / 2
            upper_percentile = lower_percentile + percentile
            lower_threshold, _, _, groups, candidate = self._calculate_polarization(
                groups, candidate, threshold, upper_percentile
            )
            upper_threshold, _, _, groups, candidate = self._calculate_polarization(
                groups, candidate, threshold, lower_percentile
            )

            if verbose:
                print(
                    f"There is a {percentile}% probability that the difference between"
                    + f" the groups' preferences for {candidate} ({groups[0]} - {groups[1]}) is"
                    + f" between [{lower_threshold:.2f}, {upper_threshold:.2f}]."
                )
            return (lower_threshold, upper_threshold)
        else:
            threshold, percentile, _, groups, candidate = self._calculate_polarization(
                groups, candidate, threshold, percentile
            )
            if verbose:
                print(
                    f"There is a {percentile:.1f}% probability that the difference between"
                    + f" the groups' preferences for {candidate} ({groups[0]} - {groups[1]}) "
                    + f" is more than {threshold:.2f}."
                )
            return percentile

    def summary(self):
        """Return a summary string for the ei results"""
        # TODO: probably format this as a table
        summary_str = """
            Computed from the raw b_ samples by multiplying by population and then 
            getting the proportion of the total pop 
            (total pop=summed across all districts):
            """
        for row in range(self.num_groups_and_num_candidates[0]):
            for col in range(self.num_groups_and_num_candidates[1]):
                summ = f"""The posterior mean for the district-level voting preference of
                {self.demographic_group_names[row]} for {self.candidate_names[col]} is
                {self.posterior_mean_voting_prefs[row][col]:.3f}
                95% equal-tailed credible interval:  {self.credible_interval_95_mean_voting_prefs[row][col]}
                """
                summary_str += summ
        return summary_str

    def precinct_level_estimates(self):
        """Returns precinct-level posterior means and credible intervals

        Returns:
        --------
            precinct_posterior_means: num_precincts x r x c
            precinct_credible_intervals: num_precincts x r x c x 2
        """

        precinct_level_samples = self.sim_trace.get_values(
            "b"
        )  # num_samples x num_precincts x r x c
        precinct_posterior_means = precinct_level_samples.mean(axis=0)
        precinct_credible_intervals = np.ones(
            (
                len(self.precinct_pops),
                self.num_groups_and_num_candidates[0],
                self.num_groups_and_num_candidates[1],
                2,
            )
        )
        percentiles = [2.5, 97.5]

        for row in range(self.num_groups_and_num_candidates[0]):
            for col in range(self.num_groups_and_num_candidates[1]):
                precinct_credible_intervals[:, row, col, :] = np.percentile(
                    precinct_level_samples[:, :, row, col], percentiles, axis=0
                ).T

        return (precinct_posterior_means, precinct_credible_intervals)

    def candidate_of_choice_report(self, verbose=True, non_candidate_names=None):
        """For each group, look at differences in preference within that group
        Parameters:
        -----------
        verbose: boolean (optional)
            If true, print a report putting the candidate preference rate in context
            If false, do not print report.
            In either case, return the candidate preference dictionary
        non_candidate_names: list of str (optional)
            A list of strings giving the names of voting outcomes that should not be
            considered candidates for the purposes of calculating the candidate of choice.
            For example, if there is an 'Abstain' column, we want may want to to include
            'Abstain' in our list of non_candidate_names so that we only consider candidates
            of choice to be actual candidates.

        Returns:
        -------
        candidate_preference_rate_dict: dict
            keys are of the form (demographic group name, candidate name)
            Values are fraction of the samples in which the support of that group for that
            candidate was higher than for any other candidate
        """

        candidate_preference_rate_dict = {}
        if non_candidate_names is None:
            non_candidate_names = []
        non_cand_idxs = [self.candidate_names.index(n) for n in non_candidate_names]
        cand_names = list(set(self.candidate_names) - set(non_candidate_names))
        sampled_voting_prefs = np.delete(self.sampled_voting_prefs, non_cand_idxs, axis=2)

        for row in range(self.num_groups_and_num_candidates[0]):
            if verbose:
                print(self.demographic_group_names[row])
            for candidate_idx, name in enumerate(cand_names):
                frac = (
                    np.argmax(sampled_voting_prefs[:, row, :], axis=1) == candidate_idx
                ).sum() / sampled_voting_prefs.shape[0]
                if verbose:
                    print(
                        f"     - In {round(frac*100,3)} percent of samples, the district-level "
                        f"vote preference of \n"
                        f"       {self.demographic_group_names[row]} for "
                        f"{name} "
                        f"was higher than for any other candidate."
                    )
                candidate_preference_rate_dict[(self.demographic_group_names[row], name)] = frac
        return candidate_preference_rate_dict

    def candidate_of_choice_polarization_report(self, verbose=True, non_candidate_names=None):
        """For each pair of groups, look at differences in preferences
        between those groups

        Parameters:
        -----------
        verbose: boolean (optional)
            If true, print a report putting the candidate of choice polarization rate context
            If false, do not pritn report.
            In either case, return the candidate difference rate dictionary
        non_candidate_names: list of str (optional)
            A list of strings giving the names of voting outcomes that should not be
            considered candidates for the purposes of calculating the candidate of choice.
            For example, if there is an 'Abstain' column, we want may want to to include
            'Abstain' in our list of non_candidate_names so that we only consider candidates
            of choice to be actual candidates.

        Returns:
        --------
        candidate_differ_rate_dict: dict
            Keys are of the form (group1, group2), values are the fraction of samples
            for which those two groups had a different candidate of choice

        Notes:
        ------
        For each pair of groups, this function reports the fraction samples for which
        the `preferred candidate` of one group (as measured by: who is the candidate supported
        by the plurality within that group according to the sampled distric-level support value)
        is different from the `preferred candidate` of the others group
        """
        candidate_differ_rate_dict = {}
        if non_candidate_names is None:
            non_candidate_names = []
        non_cand_idxs = [self.candidate_names.index(n) for n in non_candidate_names]
        sampled_voting_prefs = np.delete(self.sampled_voting_prefs, non_cand_idxs, axis=2)

        for dem1 in range(self.num_groups_and_num_candidates[0]):
            for dem2 in range(dem1):
                differ_frac = (
                    np.argmax(sampled_voting_prefs[:, dem1, :], axis=1)
                    != np.argmax(sampled_voting_prefs[:, dem2, :], axis=1)
                ).sum() / sampled_voting_prefs.shape[0]
                if verbose:
                    print(
                        f"In {round(differ_frac*100,3)} percent of samples, the district-level "
                        f"candidates of choice for {self.demographic_group_names[dem1]} and "
                        f"{self.demographic_group_names[dem2]} voters differ."
                    )
                candidate_differ_rate_dict[
                    (self.demographic_group_names[dem1], self.demographic_group_names[dem2])
                ] = differ_frac
                candidate_differ_rate_dict[
                    (self.demographic_group_names[dem2], self.demographic_group_names[dem1])
                ] = differ_frac
        return candidate_differ_rate_dict

    def plot(self):
        """Plot with no arguments returns the kde plots, with one plot for each candidate"""
        return self.plot_kdes(plot_by="candidate", axes=None)

    def plot_boxplots(self, plot_by="candidate", axes=None):
        """Plot boxplots of voting prefs (one boxplot for each candidate)

        Parameters:
        -----------
        plot_by: {'candidate', 'group'}, optional
            If candidate, make one plot for each candidate. If group, make
            one subplot for each group. (Default: 'candidate')
        axes: list of Matplotlib axis objects, optional
            Typically subplots within the same figure. Length c if plot_by = 'candidate',
            length r if plot_by = 'group'
        """
        return plot_boxplots(
            self.sampled_voting_prefs,
            self.demographic_group_names,
            self.candidate_names,
            plot_by=plot_by,
            axes=axes,
        )

    def plot_kdes(self, plot_by="candidate", axes=None):
        """Kernel density plots of voting preference, plots grouped by candidate or group

        Parameters:
        -----------
        plot_by: {'candidate', 'group'}, optional
            If candidate, make one plot for each candidate. If group, make
            one subplot for each group (Default: 'candidate')
        axes: list of Matplotlib axis objects, optional
            Typically subplots within the same figure. Length c if plot_by = 'candidate',
            length r if plot_by = 'group'
        """
        return plot_kdes(
            self.sampled_voting_prefs,
            self.demographic_group_names,
            self.candidate_names,
            plot_by=plot_by,
            axes=axes,
        )

    def plot_polarization_kde(
        self, groups, candidate, threshold=None, percentile=None, show_threshold=False, ax=None
    ):
        """Plot kde of differences between voting preferences

        Parameters
        ----------
        groups : list of strings
            Names of the demographic groups
        candidate: str
            Name of the candidate for whom the polarization is calculated
        threshold: float (optional)
            A specified level of difference in support for the candidate
            between one group and the other. If specified, use the threshold
            to calculate the percentile (exactly one of threshold and percentile
            must be None)
        percentile: float (optional)
            Between 0 and 100. Used to calculate the equal-tailed interval
            for the polarization. At least one of threshold and percentile
            must be None
        show_threshold: bool
        ax : matplotlib Axis object

        Returns
        -------
        matplotlib axis object
        """
        return_interval = threshold is None

        if return_interval:
            lower_percentile = (100 - percentile) / 2
            upper_percentile = lower_percentile + percentile
            lower_threshold, _, samples, groups, candidate = self._calculate_polarization(
                groups, candidate, threshold, upper_percentile
            )
            upper_threshold, _, samples, groups, candidate = self._calculate_polarization(
                groups, candidate, threshold, lower_percentile
            )
            thresholds = [lower_threshold, upper_threshold]
        else:
            threshold, percentile, samples, groups, candidate = self._calculate_polarization(
                groups, candidate, threshold, percentile
            )
            thresholds = [threshold]

        return plot_polarization_kde(
            samples, thresholds, percentile, groups, candidate, show_threshold, ax
        )

    def plot_intervals_by_precinct(self, group_name, candidate_name):
        """Plot of credible intervals for all precincts, for specified group and candidate

        Parameters
        ----------
        group_name : str
            Group for which to plot intervals. Should be in demographic_group_names
        candiate_name : str
            Candidate for which to plot intervals. Should be in candidate_names
        """
        if group_name not in self.demographic_group_names:
            raise ValueError(
                f"""group_name must be in the list of demographic_group_names provided to fit():
                {self.demographic_group_names}"""
            )

        if candidate_name not in self.candidate_names:
            raise ValueError(
                f"""candidate_name must be in the list of candidate_names provided to fit():
                {self.candidate_names}"""
            )

        group_index = self.demographic_group_names.index(group_name)
        candidate_index = self.candidate_names.index(candidate_name)

        point_estimates_all, intervals_all = self.precinct_level_estimates()
        point_estimates = point_estimates_all[:, group_index, candidate_index]
        intervals = intervals_all[:, group_index, candidate_index, :]

        return plot_intervals_all_precincts(
            point_estimates,
            intervals,
            candidate_name,
            self.precinct_names,
            group_name,  # TODO: _group_names_for_display?
            ax=None,
            show_all_precincts=False,
        )


def check_dimensions_of_input(
    group_fractions,
    votes_fractions,
    precinct_pops,
    demographic_group_names,
    candidate_names,
    num_groups_and_num_candidates,
):
    """Checks shape of inputs and gives warnings or errors if there is a problem

    Required arguments:
    group_fractions :   r x p (p =#precincts = num_precicts) matrix giving demographic
        information as the fraction of precinct_pop in the demographic group for each
        of p precincts and r demographic groups (sometimes denoted X)
    votes_fractions  :  c x p giving the fraction of each precinct_pop that votes
        for each of c candidates (sometimes denoted T)
    precinct_pops   :   Length-p vector giving size of each precinct population
                        of interest (e.g. voting population) (someteimes denoted N)
    Optional arguments:
    demographic_group_names  :  Names of the r demographic group of interest,
                                where results are computed for the
                                demographic group and its complement
    candidate_names          :  Name of the c candidates or voting outcomes of interest

    """

    if demographic_group_names is not None:
        if len(demographic_group_names) != num_groups_and_num_candidates[0]:
            warnings.warn(
                """Length of demographic_groups_names should be equal to
            r = group_fractions.shape[0]. If not, plotting labels may be inaccurate.
            """
            )

    if candidate_names is not None:
        if len(candidate_names) != num_groups_and_num_candidates[1]:
            warnings.warn(
                """Length of candidate_names should be equal to
            c = votes_fractions.shape[0]. If not, plotting labels be inaccurate.
            """
            )

    print(f"r = {num_groups_and_num_candidates[0]} rows (demographic groups)")
    print(f"c = {num_groups_and_num_candidates[1]} columns (candidates or voting outcomes)")
    print(f"number of precincts = {len(precinct_pops)}")

    if len(precinct_pops) != votes_fractions.shape[1]:
        raise ValueError(
            """votes_fractions should have shape: c x num_precincts.
        In particular, it is required that len(precinct_pops) = votes_fractions.shape[1]
        """
        )

    if len(precinct_pops) != group_fractions.shape[1]:
        raise ValueError(
            """votes_fractions should have shape: r x num_precincts.
        In particular, it is required that len(precinct_pops) = group_fractions.shape[1]
        """
        )
    # check shapes of group_fractions, votes_fractions, and precinct_pops to make sure
    # number of precincts match (the last dimension of each)
    if not (
        len(votes_fractions[0]) == len(group_fractions[0])
        and len(group_fractions[0]) == len(precinct_pops)
    ):
        raise ValueError(
            """Mismatching num_precincts in input shapes. Inputs should have shape: \n
        votes_fraction shape: r x num_precincts \n
        group_fractions shape: c x num_precincts \n
        precinct_pops length: num_precincts
        """
        )
