"""Models and fitting for rxc methods.

Where r and c are greater than or equal to 2.

TODO: Investigate better or reparametrized priors for multinomial-dir
TODO: Greiner-Quinn Model
TODO: Refactor to integrate with two_by_two
"""

import warnings
from typing import Any

import arviz as az
import numpy as np
import pymc as pm
from matplotlib.axes import Axes

from pyei.greiner_quinn_gibbs_sampling import pyei_greiner_quinn_sample
from pyei.plot_utils import (
    plot_boxplots,
    plot_intervals_all_precincts,
    plot_kdes,
    plot_margin_kde,
    plot_polarization_kde,
    plot_precincts,
)
from pyei.r_by_c_models import ei_multinom_dirichlet, ei_multinom_dirichlet_modified
from pyei.r_by_c_utils import check_dimensions_of_input


class RowByColumnEI:
    """Fitting and plotting for RxC models, fit via sampling"""

    def __init__(self, model_name: str, **additional_model_params: Any) -> None:  # noqa: ANN401
        """Initialize the RowByColumnEI class.

        Parameters
        ----------
        model_name: str
            The name of the model to use. Currently supported: multinomial-dirichlet,
            multinomial-dirichlet-modified
        additional_model_params: optional
            Settings for model hyperparameters, if desired to change default. See
            model function documentation for hyperparameter options
        """
        # TODO: implement greiner quinn
        # TODO: model_name as enumeration
        # TODO: clean up instance variables

        self.model_name = model_name
        self.additional_model_params = additional_model_params

        self.demographic_group_fractions: np.ndarray | None = None
        self.votes_fractions: np.ndarray | None = None
        self.precinct_pops: np.ndarray | None = None
        self.precinct_names: list[str] | None = None
        self.demographic_group_names: list[str] | None = None
        self.candidate_names: list[str] | None = None
        self.sim_model: pm.Model | None = None
        self.sim_trace: az.InferenceData | None = None
        self.sampled_voting_prefs: np.ndarray | None = None
        self.posterior_mean_voting_prefs: np.ndarray | None = None
        self.credible_interval_95_mean_voting_prefs: np.ndarray | None = None
        self.num_groups_and_num_candidates: list[int | None] = [None, None]

        self.turnout_adjusted_samples: np.ndarray | None = (
            None  # num_samples x num_precincts x r x (c-1)
        )
        self.turnout_adjusted_sampled_voting_prefs: np.ndarray | None = (
            None  # samps districtwide prefs,num_samples x r x c-1
        )
        self.turnout_adjusted_candidate_names: list[str] | None = (
            None  # candidate names with no-vote column omitted
        )
        self.turnout_adjusted_posterior_mean_voting_prefs: np.ndarray | None = None
        self.turnout_adjusted_credible_interval_95_mean_voting_prefs: (
            np.ndarray | None
        ) = None
        self.turnout_samples: np.ndarray | None = None

    def fit(
        self,
        group_fractions: np.ndarray,
        votes_fractions: np.ndarray,
        precinct_pops: np.ndarray,
        demographic_group_names: list[str] | None = None,
        candidate_names: list[str] | None = None,
        target_accept: float = 0.99,
        tune: int = 1500,
        draw_samples: bool = True,
        precinct_names: list[str] | None = None,
        **other_sampling_args: Any,  # noqa: ANN401
    ) -> None:
        """Fit the specified model using MCMC sampling.

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
        if not all(isinstance(p, int | np.integer) for p in precinct_pops):
            raise ValueError("all elements of precinct_pops must be integer-valued")
        self.precinct_pops = precinct_pops

        # check that group_fractions and vote_fractions sum to 1 in each precinct
        if not np.isclose(group_fractions.sum(axis=0), 1.0).all():
            raise ValueError("group_fractions should sum to 1 within each precinct")
        if not np.isclose(votes_fractions.sum(axis=0), 1.0).all():
            raise ValueError("votes_fractions should sum to 1 within each precinct")

        # check that group_fractions and vote_fractions are nonnegative
        if not (group_fractions >= 0).all():
            raise ValueError("group_fractions must be non-negative")
        if not (votes_fractions >= 0).all():
            raise ValueError("votes_fractions most be non-negative")

        # give demographic groups, candidates 1-indexed numbers as names if names are not specified
        if demographic_group_names is None:
            demographic_group_names = [
                str(i) for i in range(1, group_fractions.shape[0] + 1)
            ]
        if candidate_names is None:
            candidate_names = [str(i) for i in range(1, votes_fractions.shape[0] + 1)]
        self.demographic_group_names = demographic_group_names
        self.candidate_names = candidate_names

        # Set precinct names
        if precinct_names is not None:
            if len(precinct_names) != len(precinct_pops):
                raise ValueError(
                    "precinct_names and precinct_pops must have the same length"
                )
            if len(set(precinct_names)) != len(precinct_names):
                warnings.warn(
                    "Precinct names are not unique. This may interfere with "
                    "passing precinct names to precinct_level_plot().",
                    stacklevel=2,
                )
            self.precinct_names = list(precinct_names)

        self.num_groups_and_num_candidates = [
            group_fractions.shape[0],
            votes_fractions.shape[0],
        ]  # [r, c]

        # Type narrowing for mypy
        num_groups = self.num_groups_and_num_candidates[0]
        num_candidates = self.num_groups_and_num_candidates[1]
        if num_groups is None or num_candidates is None:
            raise ValueError("num_groups_and_num_candidates must be set")

        check_dimensions_of_input(
            group_fractions,
            votes_fractions,
            precinct_pops,
            demographic_group_names,
            candidate_names,
            (num_groups, num_candidates),
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

        elif self.model_name == "greiner-quinn":
            self.sim_model = None

        else:
            raise ValueError(
                f"""{self.model_name} is not a supported model_name
            Currently supported: RxC models: 'multinomial-dirichlet-modified',
            'multinomial-dirichlet' """
            )

        if draw_samples:
            if self.model_name in [
                "multinomial-dirichlet-modified",
                "multinomial-dirichlet",
            ]:  # for models whose sampling is w/ pycm
                if self.sim_model is None:
                    raise ValueError("sim_model must be set before sampling")
                with self.sim_model:
                    self.sim_trace = pm.sample(
                        target_accept=target_accept,
                        tune=tune,
                        nuts_sampler="numpyro",
                        **other_sampling_args,
                    )
            elif self.model_name == "greiner-quinn":
                self.sim_trace = pyei_greiner_quinn_sample(
                    group_fractions,
                    votes_fractions,
                    precinct_pops,
                    **other_sampling_args,
                )  #

            self.calculate_summary()

    def _calculate_turnout_adjusted_samples(
        self, non_candidate_names: list[str]
    ) -> None:
        """Calculate voting support as a fraction of all those who voted.

        For each sample, calculate the voting support of each group for each candidate
        *as a fraction of all those who voted* (instead of as a fraction of all those
        included in the precinct population. This fn is only applicable when one of the
        c voting outcomes is a no-vote or abstain column. In this case, the total number
        of voters is unknown but samples from its distribution can be calculated).

        Parameters
        ----------
        non_candidate_names: list of str
            each a name of the column/ voting outcome that corresponds to not voting,
            if applicable. Each string in the list must be in candidate_names

        Notes:
        -----
        Sets the values of:
            self.turnout_adjusted_candidate_names
            self.turnout_samples
            self.turnout_adjusted_samples
        """
        if self.candidate_names is None:
            raise ValueError(
                "candidate_names must be set before calling _calculate_turnout_adjusted_samples"
            )
        if self.sim_trace is None:
            raise ValueError(
                "sim_trace must be set before calling _calculate_turnout_adjusted_samples"
            )

        abstain_column_indices = []
        for non_candidate_name in non_candidate_names:
            if non_candidate_name not in self.candidate_names:
                raise ValueError(
                    f"non_candidate_names must be in candidate_names: {self.candidate_names}"
                )
            abstain_column_indices.append(
                self.candidate_names.index(non_candidate_name)
            )

        non_adjusted_samples = np.transpose(
            self.sim_trace["posterior"]["b"].stack(all_draws=["chain", "draw"]).values,  # noqa: PD013
            axes=(3, 0, 1, 2),
        )  # num_samples x num_precincts x r x c  # num_samples x num_precincts x r x c

        self.turnout_adjusted_candidate_names = [
            name for name in self.candidate_names if name not in non_candidate_names
        ]

        total_abstentions = non_adjusted_samples[:, :, :, abstain_column_indices].sum(
            axis=3
        )  # total fraction in all vote columnn num_samples x num_precincts x r

        if self.demographic_group_fractions is None or self.precinct_pops is None:
            raise ValueError(
                "demographic_group_fractions and precinct_pops must be set before calling _calculate_turnout_adjusted_samples"
            )
        self.turnout_samples = (
            1 - total_abstentions  # fraction that aren't in the no-vote column(s)
        ) * np.swapaxes(self.demographic_group_fractions * self.precinct_pops, 0, 1)

        turnout_adjusted_samples = np.delete(
            non_adjusted_samples, abstain_column_indices, axis=3
        )  # num_samples x num_precincts x r x c-1

        self.turnout_adjusted_samples = (
            turnout_adjusted_samples
            / turnout_adjusted_samples.sum(axis=3, keepdims=True)
        )  # num_samples x num_precincts x r x c-1

    def calculate_turnout_adjusted_summary(
        self, non_candidate_names: list[str]
    ) -> None:
        """Calculates districtwide samples, means, and credible intervals

        Parameters
        ----------
        non_candidate_names: list of str
            each a name of the column/ voting outcome that corresponds to not voting,
            if applicable. Each string in the list must be in candidate_names

        Notes:
        -----
        Sets turnout_adjusted_voting_prefs, turnout_adjusted_posterior_mean_voting_prefs,
            turnout_adjusted_credible_interval_95_mean_voting_prefs
        """
        self._calculate_turnout_adjusted_samples(non_candidate_names)

        if self.turnout_adjusted_samples is None or self.turnout_samples is None:
            raise ValueError(
                "turnout_adjusted_samples and turnout_samples must be set before calling calculate_turnout_adjusted_summary"
            )
        samples_converted_to_pops = (
            np.transpose(self.turnout_adjusted_samples, axes=(3, 0, 1, 2))
            * self.turnout_samples
        )
        # (c-1) x num_samples x num_precincts x r x
        samples_of_votes_summed_across_district = samples_converted_to_pops.sum(
            axis=2
        )  # (c-1) x num_samples x r

        # obtain samples of the districtwide proportion of each demog. group voting for candidate
        self.turnout_adjusted_sampled_voting_prefs = np.transpose(
            samples_of_votes_summed_across_district / self.turnout_samples.sum(axis=1),
            axes=(1, 2, 0),
        )  # sampled voted prefs across precincts,  num_samples x r x c-1

        # # compute point estimates
        self.turnout_adjusted_posterior_mean_voting_prefs = (
            self.turnout_adjusted_sampled_voting_prefs.mean(axis=0)
        )  # r x (c -1)

        # # compute credible intervals
        percentiles = [2.5, 97.5]
        if (
            self.num_groups_and_num_candidates[0] is None
            or self.num_groups_and_num_candidates[1] is None
        ):
            raise ValueError(
                "num_groups_and_num_candidates must be set before calling calculate_turnout_adjusted_summary"
            )

        self.turnout_adjusted_credible_interval_95_mean_voting_prefs = np.zeros(
            (
                self.num_groups_and_num_candidates[0],
                self.num_groups_and_num_candidates[1] - 1,
                2,
            )
        )
        for row in range(self.num_groups_and_num_candidates[0]):
            for col in range(self.num_groups_and_num_candidates[1] - 1):
                self.turnout_adjusted_credible_interval_95_mean_voting_prefs[row][col][
                    :
                ] = np.percentile(
                    self.turnout_adjusted_sampled_voting_prefs[:, row, col], percentiles
                )

    def calculate_summary(self) -> None:
        """Calculate point estimates (post. means) and 95% equal-tailed credible intervals

        Sets
            self.sampled_voting_prefs
            self.posterior_mean_voting_prefs
            self.credible_interval_95_mean_voting_prefs
        """
        # multiply sample proportions by precinct pops for each group to get samples of
        # number of voters of the demographic group who voted for the candidate
        # in each precinct
        if self.sim_trace is None:
            raise ValueError("sim_trace must be set before calling calculate_summary")
        if self.demographic_group_fractions is None or self.precinct_pops is None:
            raise ValueError(
                "demographic_group_fractions and precinct_pops must be set before calling calculate_summary"
            )
        # This next messy line created to extract/reshape the InferenceData object to
        # match what was previously returned by self.sim_trace.get_values("b")
        # (needs to flatten out the chains dimension)
        b_values = np.transpose(
            self.sim_trace["posterior"]["b"].stack(all_draws=["chain", "draw"]).values,  # noqa: PD013
            axes=(3, 0, 1, 2),
        )  # num_samples x num_precincts x r x c
        demographic_group_counts = np.transpose(
            self.demographic_group_fractions * self.precinct_pops
        )  # num_precincts x r
        samples_converted_to_pops = (
            np.transpose(b_values, axes=(3, 0, 1, 2)) * demographic_group_counts
        )
        # c x num_samples x num_precincts x r

        samples_of_votes_summed_across_district = samples_converted_to_pops.sum(
            axis=2
        )  # c x num_samples x r

        # obtain samples of the districtwide proportion of each demog. group voting for candidate
        self.sampled_voting_prefs = np.transpose(
            samples_of_votes_summed_across_district
            / demographic_group_counts.sum(axis=0),
            axes=(1, 2, 0),
        )  # sampled voted prefs across precincts,  num_samples x r x c

        # # compute point estimates
        self.posterior_mean_voting_prefs = self.sampled_voting_prefs.mean(
            axis=0
        )  # r x c

        # compute credible intervals
        percentiles = [2.5, 97.5]
        self.credible_interval_95_mean_voting_prefs = np.zeros(
            (
                self.num_groups_and_num_candidates[0],
                self.num_groups_and_num_candidates[1],
                2,
            )
        )
        if (
            self.num_groups_and_num_candidates[0] is None
            or self.num_groups_and_num_candidates[1] is None
        ):
            raise ValueError(
                "num_groups_and_num_candidates must be set before calling calculate_summary"
            )
        for row in range(self.num_groups_and_num_candidates[0]):
            for col in range(self.num_groups_and_num_candidates[1]):
                self.credible_interval_95_mean_voting_prefs[row][col][:] = (
                    np.percentile(self.sampled_voting_prefs[:, row, col], percentiles)
                )

    def _calculate_margin(
        self,
        group: int,
        candidates: list[int],
        threshold: float | None = None,
        percentile: float | None = None,
    ) -> float:
        """Calculate the Candidate 1 - Candidate 2 margin among the given group.

        Calculate the percentile given a threshold, or vice versa. Exactly one of
        {percentile, threshold} must be None.

        Parameters:
        ----------
        group: str
            Demographic group in question
        candidates: list of str
            Length 2 vector of candidates upon which to calculate the margin
        threshold: float (optional)
            A specified level for the margin between the two candidates. If specified,
            use the threshold to calculate the percentile (% of samples with a larger margin)
        percentile: float (opetional)
            Between 0 and 100. Used to calculate the equal-tailed interval for the margin between
            the two candidates.

        Returns:
        --------
            threshold
            percentile
            samples
            group
            candidates
        """
        if self.candidate_names is None or self.demographic_group_names is None:
            raise ValueError(
                "candidate_names and demographic_group_names must be set before calling _calculate_margin"
            )
        if self.sampled_voting_prefs is None:
            raise ValueError(
                "sampled_voting_prefs must be set before calling _calculate_margin"
            )

        # TODO: document return values
        candidate_index_0 = self.candidate_names.index(str(candidates[0]))
        candidate_index_1 = self.candidate_names.index(str(candidates[1]))
        group_index = self.demographic_group_names.index(str(group))

        samples = (
            self.sampled_voting_prefs[:, group_index, candidate_index_0]
            - self.sampled_voting_prefs[:, group_index, candidate_index_1]
        )
        if percentile is None and threshold is not None:
            percentile = (
                100 * (samples > threshold).sum() / len(self.sampled_voting_prefs)
            )
        elif threshold is None and percentile is not None:
            threshold = np.percentile(samples, 100 - percentile)
        else:
            raise ValueError(
                """Exactly one of threshold or percentile must be None.
            Set a threshold to calculate the associated percentile, or a percentile
            to calculate the associated threshold.
            """
            )
        return threshold

    def margin_report(
        self,
        group: int,
        candidates: list[int],
        threshold: float | None = None,
        percentile: float | None = None,
        verbose: bool = True,
    ) -> str:
        """Calculate margin probability or confidence interval.

        For a given threshold, return the probability that the margin between
        the two candidates preferences in the given demographic group is greater than
        the threshold OR for a given confidence level, calculate the associated confidence interval
        of the difference between the two candidates preference among the group.
        Exactly one of {percentile, threshold} must be None.

        Parameters:
        -----------
        group: str
            Demographic group in question
        candidates: list of str
            Length 2 vector of candidates upon which to calculate the margin
        threshold: float (optional)
            A specified level for the margin between the two candidates. If specified,
            use the threshold to calculate the percentile (% of samples with a larger margin)
        percentile: float (opetional)
            Between 0 and 100. Used to calculate the equal-tailed interval for the margin between
            the two candidates.
        verbose: bool
            If true, print a report putting margin in context
        """
        return_interval = threshold is None

        if self.candidate_names is None or self.demographic_group_names is None:
            raise ValueError(
                "candidate_names and demographic_group_names must be set before calling margin_report"
            )
        if not all(candidate in self.candidate_names for candidate in candidates):
            raise ValueError(
                f"""candidate names must be in the list of candidate_names provided to fit():
                {self.candidate_names}"""
            )

        if group not in self.demographic_group_names:
            raise ValueError(
                f"""group name must be in the list of demographic_group_names
                provided to fit():
                {self.demographic_group_names}"""
            )

        if return_interval:
            if percentile is None:
                raise ValueError("percentile must be provided when threshold is None")
            lower_percentile = (100 - percentile) / 2
            upper_percentile = lower_percentile + percentile
            lower_threshold = self._calculate_margin(
                group, candidates, threshold, upper_percentile
            )
            upper_threshold = self._calculate_margin(
                group, candidates, threshold, lower_percentile
            )

            if verbose:
                print(
                    f"There is a {percentile}% probability that the difference between"
                    + f" {group}s' preferences for {candidates[0]} and {candidates[1]} is"
                    + f" between [{lower_threshold:.2f}, {upper_threshold:.2f}]."
                )
            return f"({lower_threshold:.2f}, {upper_threshold:.2f})"
        else:
            if threshold is None:
                raise ValueError("threshold must be provided when percentile is None")
            threshold = self._calculate_margin(group, candidates, threshold, percentile)
            if verbose:
                print(
                    f"There is a {percentile:.1f}% probability that the difference between"
                    + f" {group}s' preferences for {candidates[0]} and {candidates[1]}"
                    + f" is more than {threshold:.2f}."
                )
            return str(threshold)

    def _calculate_polarization(
        self,
        groups: list[int],
        candidate: int,
        threshold: float | None = None,
        percentile: float | None = None,
    ) -> float:
        """Calculate percentile given a threshold, or vice versa.

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
        if self.candidate_names is None or self.demographic_group_names is None:
            raise ValueError(
                "candidate_names and demographic_group_names must be set before calling _calculate_polarization"
            )
        candidate_index = self.candidate_names.index(str(candidate))
        group_index_0 = self.demographic_group_names.index(str(groups[0]))
        group_index_1 = self.demographic_group_names.index(str(groups[1]))

        if self.sampled_voting_prefs is None:
            raise ValueError(
                "sampled_voting_prefs must be set before calling _calculate_polarization"
            )
        samples = (
            self.sampled_voting_prefs[:, group_index_0, candidate_index]
            - self.sampled_voting_prefs[:, group_index_1, candidate_index]
        )

        if percentile is None and threshold is not None:
            percentile = (
                100 * (samples > threshold).sum() / len(self.sampled_voting_prefs)
            )
        elif threshold is None and percentile is not None:
            threshold = np.percentile(samples, 100 - percentile)
        else:
            raise ValueError(
                """Exactly one of threshold or percentile must be None.
            Set a threshold to calculate the associated percentile, or a percentile
            to calculate the associated threshold.
            """
            )
        return threshold

    def polarization_report(
        self,
        groups: list[int],
        candidate: int,
        threshold: float | None = None,
        percentile: float | None = None,
        verbose: bool = True,
    ) -> float | str:
        """For a given threshold, return the probability that the difference between

        the two demographic groups' preferences for the candidate is greater than
        the threshold OR For a given confidence level, calculate the associated
        confidence interval of the difference between the two groups' preferences.
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

        if self.demographic_group_names is None or self.candidate_names is None:
            raise ValueError(
                "demographic_group_names and candidate_names must be set before calling polarization_report"
            )
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
            if percentile is None:
                raise ValueError("percentile must be provided when threshold is None")
            lower_percentile = (100 - percentile) / 2
            upper_percentile = lower_percentile + percentile
            lower_threshold = self._calculate_polarization(
                groups, candidate, threshold, upper_percentile
            )
            upper_threshold = self._calculate_polarization(
                groups, candidate, threshold, lower_percentile
            )

            if verbose:
                print(
                    f"There is a {percentile}% probability that the difference between"
                    + f" the groups' preferences for {candidate} ({groups[0]} - {groups[1]}) is"
                    + f" between [{lower_threshold:.2f}, {upper_threshold:.2f}]."
                )
            return (lower_threshold, upper_threshold)  # type: ignore[return-value]
        else:
            if threshold is None:
                raise ValueError("threshold must be provided when percentile is None")
            threshold = self._calculate_polarization(
                groups, candidate, threshold, percentile
            )
            if verbose:
                # Calculate the actual percentile from the threshold
                if (
                    self.sampled_voting_prefs is None
                    or self.candidate_names is None
                    or self.demographic_group_names is None
                ):
                    raise ValueError(
                        "sampled_voting_prefs, candidate_names, and demographic_group_names must be set"
                    )
                candidate_index = self.candidate_names.index(str(candidate))
                group_index_0 = self.demographic_group_names.index(str(groups[0]))
                group_index_1 = self.demographic_group_names.index(str(groups[1]))
                samples = (
                    self.sampled_voting_prefs[:, group_index_0, candidate_index]
                    - self.sampled_voting_prefs[:, group_index_1, candidate_index]
                )
                actual_percentile = 100 * (samples > threshold).sum() / len(samples)
                print(
                    f"There is a {actual_percentile:.1f}% probability that the difference between"
                    + f" the groups' preferences for {candidate} ({groups[0]} - {groups[1]}) "
                    + f" is more than {threshold:.2f}."
                )
            return actual_percentile / 100.0

    def summary(self, non_candidate_names: list[str] | None = None) -> str:
        """Return a summary string for the ei results

        Parameters:
        -----------
        non_candidate_names: list of str (optional)
            Each a string in self.candidate_names() that corresponds to the name of a "no-vote"
            or abstain column, if applicable. If passed, the summary will be estimates of
            support for candidates AMONG those who were estimated to not be in the
            specified abstain column
        """
        # TODO: probably format this as a table
        summary_str = """
            Computed from the raw b_ samples by multiplying by group populations and then
            getting the proportion of the total pop
            (total pop=summed across all districts):
            """
        if non_candidate_names is not None:
            self.calculate_turnout_adjusted_summary(non_candidate_names)
            candidate_names_for_summary = self.turnout_adjusted_candidate_names
            posterior_means = self.turnout_adjusted_posterior_mean_voting_prefs
            credible_intervals = (
                self.turnout_adjusted_credible_interval_95_mean_voting_prefs
            )
        else:
            candidate_names_for_summary = self.candidate_names
            posterior_means = self.posterior_mean_voting_prefs
            credible_intervals = self.credible_interval_95_mean_voting_prefs

        if (
            self.num_groups_and_num_candidates[0] is None
            or self.demographic_group_names is None
        ):
            raise ValueError(
                "num_groups_and_num_candidates and demographic_group_names must be set before calling summary"
            )
        for row in range(self.num_groups_and_num_candidates[0]):
            for col, candidate_name in enumerate(candidate_names_for_summary or []):
                if posterior_means is not None and credible_intervals is not None:
                    summ = f"""The posterior mean for the district-level voting preference of
                    {self.demographic_group_names[row]} for {candidate_name} is
                    {posterior_means[row][col]:.3f}
                    95% equal-tailed credible interval:  {credible_intervals[row][col]}
                    """
                else:
                    summ = ""
                summary_str += summ
        return summary_str

    def precinct_level_estimates(
        self, non_candidate_names: list[str] | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Returns precinct-level posterior means and credible intervals

        Parameters:
        ----------
        non_candidate_names: list of str
            Optional. If specified, this will give the names of column to be
            treated as no-vote columns, and the precinct-level estimates
            will be computed AMONG those who were estimated to have voted

        Returns:
        --------
            precinct_posterior_means: num_precincts x r x c
            precinct_credible_intervals: num_precincts x r x c x 2
        """
        if self.sim_trace is None:
            raise ValueError(
                "sim_trace must be set before calling precinct_level_estimates"
            )
        if non_candidate_names is not None:
            if self.turnout_adjusted_samples is None:
                raise ValueError(
                    "turnout_adjusted_samples must be set before calling precinct_level_estimates"
                )
            precinct_level_samples = self.turnout_adjusted_samples
        else:
            precinct_level_samples = np.transpose(
                self.sim_trace["posterior"]["b"]  # noqa: PD013
                .stack(all_draws=["chain", "draw"])
                .values,
                axes=(3, 0, 1, 2),
            )  # num_samples x num_precincts x r x c # num_samples x num_precincts x r x c
        if self.precinct_pops is None:
            raise ValueError(
                "precinct_pops must be set before calling precinct_level_estimates"
            )

        _, _, r, c = precinct_level_samples.shape
        precinct_posterior_means = precinct_level_samples.mean(axis=0)
        precinct_credible_intervals = np.ones(
            (
                len(self.precinct_pops),
                r,
                c,
                2,
            )
        )
        percentiles = [2.5, 97.5]

        for row in range(r):
            for col in range(c):
                precinct_credible_intervals[:, row, col, :] = np.percentile(
                    precinct_level_samples[:, :, row, col], percentiles, axis=0
                ).T

        return (precinct_posterior_means, precinct_credible_intervals)

    def candidate_of_choice_report(
        self, verbose: bool = True, non_candidate_names: list[str] | None = None
    ) -> dict[tuple[str, str], float]:
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
        if self.candidate_names is None:
            raise ValueError(
                "candidate_names must be set before calling candidate_of_choice_report"
            )
        if self.sampled_voting_prefs is None:
            raise ValueError(
                "sampled_voting_prefs must be set before calling candidate_of_choice_report"
            )
        candidate_preference_rate_dict = {}
        if non_candidate_names is None:
            non_candidate_names = []
        non_cand_idxs = [self.candidate_names.index(n) for n in non_candidate_names]
        cand_names = [c for c in self.candidate_names if c not in non_candidate_names]
        sampled_voting_prefs = np.delete(
            self.sampled_voting_prefs, non_cand_idxs, axis=2
        )

        if (
            self.num_groups_and_num_candidates[0] is None
            or self.demographic_group_names is None
        ):
            raise ValueError(
                "num_groups_and_num_candidates and demographic_group_names must be set before calling candidate_of_choice_report"
            )
        for row in range(self.num_groups_and_num_candidates[0]):
            if verbose:
                print(self.demographic_group_names[row])
            for candidate_idx, name in enumerate(cand_names):
                frac = (
                    np.argmax(sampled_voting_prefs[:, row, :], axis=1) == candidate_idx
                ).sum() / sampled_voting_prefs.shape[0]
                if verbose:
                    print(
                        f"     - In {round(frac * 100, 3)} percent of samples, the district-level "
                        f"vote preference of \n"
                        f"       {self.demographic_group_names[row]} for "
                        f"{name} "
                        f"was higher than for any other candidate."
                    )
                candidate_preference_rate_dict[
                    (self.demographic_group_names[row], name)
                ] = frac
        return candidate_preference_rate_dict

    def candidate_of_choice_polarization_report(
        self, verbose: bool = True, non_candidate_names: list[str] | None = None
    ) -> dict[tuple[str, str], float]:
        """For each pair of groups, look at differences in preferences between those groups

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
        if self.candidate_names is None:
            raise ValueError(
                "candidate_names must be set before calling candidate_of_choice_polarization_report"
            )
        if self.sampled_voting_prefs is None:
            raise ValueError(
                "sampled_voting_prefs must be set before calling candidate_of_choice_polarization_report"
            )
        candidate_differ_rate_dict = {}
        if non_candidate_names is None:
            non_candidate_names = []
        non_cand_idxs = [self.candidate_names.index(n) for n in non_candidate_names]
        sampled_voting_prefs = np.delete(
            self.sampled_voting_prefs, non_cand_idxs, axis=2
        )

        if (
            self.num_groups_and_num_candidates[0] is None
            or self.demographic_group_names is None
        ):
            raise ValueError(
                "num_groups_and_num_candidates and demographic_group_names must be set before calling candidate_of_choice_polarization_report"
            )
        for dem1 in range(self.num_groups_and_num_candidates[0]):
            for dem2 in range(dem1):
                differ_frac = (
                    np.argmax(sampled_voting_prefs[:, dem1, :], axis=1)
                    != np.argmax(sampled_voting_prefs[:, dem2, :], axis=1)
                ).sum() / sampled_voting_prefs.shape[0]
                if verbose:
                    print(
                        f"In {round(differ_frac * 100, 3)} percent of samples, the district-level "
                        f"candidates of choice for {self.demographic_group_names[dem1]} and "
                        f"{self.demographic_group_names[dem2]} voters differ."
                    )
                candidate_differ_rate_dict[
                    (
                        self.demographic_group_names[dem1],
                        self.demographic_group_names[dem2],
                    )
                ] = differ_frac
                candidate_differ_rate_dict[
                    (
                        self.demographic_group_names[dem2],
                        self.demographic_group_names[dem1],
                    )
                ] = differ_frac
        return candidate_differ_rate_dict

    def plot(
        self, non_candidate_names: list[str] | None = None
    ) -> Axes | tuple[Axes, ...]:
        """Plot with no arguments returns the kde plots, with one plot for each candidate

        Parameters:
        non_candidate_names: list of str
            each a name of the column/ voting outcome that corresponds to not voting,
            if applicable. Each string in the list must be in candidate_names
        """
        return self.plot_kdes(
            plot_by="candidate", non_candidate_names=non_candidate_names, axes=None
        )

    def plot_boxplots(
        self,
        plot_by: str = "candidate",
        non_candidate_names: list[str] | None = None,
        axes: Axes | None = None,
    ) -> Axes:
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
        if self.sampled_voting_prefs is None or self.candidate_names is None:
            raise ValueError(
                "sampled_voting_prefs and candidate_names must be set before calling plot_boxplots"
            )
        if non_candidate_names is None:
            voting_prefs = self.sampled_voting_prefs
            candidate_names = self.candidate_names
        else:  # turnout adjusted samples, names without no-vote column
            self.calculate_turnout_adjusted_summary(non_candidate_names)
            if (
                self.turnout_adjusted_sampled_voting_prefs is None
                or self.turnout_adjusted_candidate_names is None
            ):
                raise ValueError(
                    "turnout_adjusted_sampled_voting_prefs and turnout_adjusted_candidate_names must be set"
                )
            voting_prefs = self.turnout_adjusted_sampled_voting_prefs
            candidate_names = self.turnout_adjusted_candidate_names

        if self.demographic_group_names is None:
            raise ValueError("demographic_group_names must be set")

        return plot_boxplots(
            voting_prefs,
            self.demographic_group_names,
            candidate_names,
            plot_by=plot_by,
            axes=axes,
        )

    def plot_kdes(
        self,
        plot_by: str = "candidate",
        non_candidate_names: list[str] | None = None,
        axes: Axes | None = None,
    ) -> Axes:
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
        if self.sampled_voting_prefs is None or self.candidate_names is None:
            raise ValueError(
                "sampled_voting_prefs and candidate_names must be set before calling plot_kdes"
            )
        if non_candidate_names is None:
            voting_prefs = self.sampled_voting_prefs
            candidate_names = self.candidate_names
        else:  # turnout adjusted samples, names without no-vote column
            self.calculate_turnout_adjusted_summary(non_candidate_names)
            if (
                self.turnout_adjusted_sampled_voting_prefs is None
                or self.turnout_adjusted_candidate_names is None
            ):
                raise ValueError(
                    "turnout_adjusted_sampled_voting_prefs and turnout_adjusted_candidate_names must be set"
                )
            voting_prefs = self.turnout_adjusted_sampled_voting_prefs
            candidate_names = self.turnout_adjusted_candidate_names

        if self.demographic_group_names is None:
            raise ValueError("demographic_group_names must be set")

        return plot_kdes(
            voting_prefs,
            self.demographic_group_names,
            candidate_names,
            plot_by=plot_by,
            axes=axes,
        )

    def plot_margin_kde(
        self,
        group: int,
        candidates: list[int],
        threshold: float | None = None,
        percentile: float | None = None,
        show_threshold: bool = False,
        ax: Axes | None = None,
    ) -> Axes:
        """Plot kde of the margin between two candidates among the given demographic group.

        Parameters:
        ----------
        group: str
            Demographic group in question
        candidates: list of str
            Length 2 vector of candidates upon which to calculate the margin
        threshold: float (optional)
            A specified level for the margin between the two candidates. If specified,
            use the threshold to calculate the percentile (% of samples with a larger margin)
        percentile: float (opetional)
            Between 0 and 100. Used to calculate the equal-tailed interval for the margin between
            the two candidates.
        show_threshold: bool
            Show threshold in the plot.
        """
        return_interval = threshold is None
        if return_interval:
            if percentile is None:
                raise ValueError("percentile must be provided when threshold is None")
            lower_percentile = (100 - percentile) / 2
            upper_percentile = lower_percentile + percentile
            lower_threshold = self._calculate_margin(
                group, candidates, threshold, upper_percentile
            )
            upper_threshold = self._calculate_margin(
                group, candidates, threshold, lower_percentile
            )
            thresholds = [lower_threshold, upper_threshold]
        else:
            if threshold is None:
                raise ValueError("threshold must be provided when percentile is None")
            threshold = self._calculate_margin(group, candidates, threshold, percentile)
            thresholds = [threshold]

        # Get samples for plotting
        if (
            self.sampled_voting_prefs is None
            or self.candidate_names is None
            or self.demographic_group_names is None
        ):
            raise ValueError(
                "sampled_voting_prefs, candidate_names, and demographic_group_names must be set"
            )

        candidate_index_0 = self.candidate_names.index(str(candidates[0]))
        candidate_index_1 = self.candidate_names.index(str(candidates[1]))
        group_index = self.demographic_group_names.index(str(group))
        samples = (
            self.sampled_voting_prefs[:, group_index, candidate_index_0]
            - self.sampled_voting_prefs[:, group_index, candidate_index_1]
        )

        from matplotlib.figure import Figure

        if ax is None:
            fig = Figure()
            ax = fig.add_subplot(111)

        return plot_margin_kde(
            str(group),
            [str(c) for c in candidates],
            samples,
            thresholds,
            percentile or 0.0,
            show_threshold,
            ax,
        )

    def plot_polarization_kde(
        self,
        groups: list[int],
        candidate: int,
        threshold: float | None = None,
        percentile: float | None = None,
        show_threshold: bool = False,
        ax: Axes | None = None,
        color: str = "steelblue",
    ) -> Axes:
        """Plot kde of differences between voting preferences

        Parameters:
        -----------
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
        color : str (optional)
            Specifies a color for matplotlib to be used in the histogram/kde.
            default="steelblue"

        Returns:
        --------
        matplotlib axis object
        """
        return_interval = threshold is None

        if return_interval:
            if percentile is None:
                raise ValueError("percentile must be provided when threshold is None")
            lower_percentile = (100 - percentile) / 2
            upper_percentile = lower_percentile + percentile
            lower_threshold = self._calculate_polarization(
                groups, candidate, threshold, upper_percentile
            )
            upper_threshold = self._calculate_polarization(
                groups, candidate, threshold, lower_percentile
            )
            thresholds = [lower_threshold, upper_threshold]
        else:
            if threshold is None:
                raise ValueError("threshold must be provided when percentile is None")
            threshold = self._calculate_polarization(
                groups, candidate, threshold, percentile
            )
            thresholds = [threshold]

        # Get samples for plotting
        if (
            self.sampled_voting_prefs is None
            or self.candidate_names is None
            or self.demographic_group_names is None
        ):
            raise ValueError(
                "sampled_voting_prefs, candidate_names, and demographic_group_names must be set"
            )

        candidate_index = self.candidate_names.index(str(candidate))
        group_index_0 = self.demographic_group_names.index(str(groups[0]))
        group_index_1 = self.demographic_group_names.index(str(groups[1]))
        samples = (
            self.sampled_voting_prefs[:, group_index_0, candidate_index]
            - self.sampled_voting_prefs[:, group_index_1, candidate_index]
        )

        return plot_polarization_kde(
            samples,
            thresholds,
            percentile or 0.0,
            [str(g) for g in groups],
            str(candidate),
            show_threshold,
            ax,
            color=color,
        )

    def plot_intervals_by_precinct(self, group_name: str, candidate_name: str) -> Axes:
        """Plot of credible intervals for all precincts, for specified group and candidate

        Parameters:
        -----------
        group_name : str
            Group for which to plot intervals. Should be in demographic_group_names
        candiate_name : str
            Candidate for which to plot intervals. Should be in candidate_names
        """
        if self.demographic_group_names is None or self.candidate_names is None:
            raise ValueError(
                "demographic_group_names and candidate_names must be set before calling plot_intervals_by_precinct"
            )
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

        if self.precinct_names is None:
            raise ValueError(
                "precinct_names must be set before calling plot_intervals_by_precinct"
            )
        return plot_intervals_all_precincts(
            point_estimates,
            intervals,
            candidate_name,
            self.precinct_names,
            group_name,  # TODO: _group_names_for_display?
            ax=None,
            show_all_precincts=False,
        )

    def precinct_level_plot(
        self,
        candidate: int,
        groups: list[str] | None = None,
        alpha: float = 1,
        ax: Axes | None = None,
        show_all_precincts: bool = False,
        precinct_names: list[str] | None = None,
        plot_as_histograms: bool = False,
    ) -> Axes:
        """Plot ridgeplots for precincts.

        Parameters
        ----------
        candidate : str, optional
            The candidate whose support we're examining
        groups : list of str, optional
            The groups whose support we're examining
        alpha : float, optional
            The opacity of the ridgeplots' fill color
        ax : matplotlib axes object, optional
            The axes to plot on
        show_all_precincts : bool, optional
            If True, then it will show all ridge plots (even if there are more than 50)
        precinct_names : list[str], optional
            Labels for each precinct (if not supplied, by default we label each precinct with an integer label, 1 to n)
        plot_as_histograms : bool, optional
            Default is false. If true, plot with histograms instead of kdes
        """
        if self.sim_trace is None:
            raise ValueError("sim_trace must be set before calling precinct_level_plot")
        if self.demographic_group_names is None or self.candidate_names is None:
            raise ValueError(
                "demographic_group_names and candidate_names must be set before calling precinct_level_plot"
            )
        precinct_level_samples = np.transpose(
            self.sim_trace["posterior"]["b"].stack(all_draws=["chain", "draw"]).values,  # noqa: PD013
            axes=(3, 0, 1, 2),
        )  # num_samples x num_precincts x r x c  # num_samples x num_precincts x r x c
        if groups is None:
            if self.demographic_group_names is None:
                raise ValueError("demographic_group_names must be set")
            groups = self.demographic_group_names

        candidate_idx = self.candidate_names.index(str(candidate))
        voting_prefs = []
        for group in groups:
            group_idx = self.demographic_group_names.index(str(group))
            voting_prefs.append(precinct_level_samples[:, :, group_idx, candidate_idx])
        return plot_precincts(
            voting_prefs,
            group_names=groups,
            candidate=str(candidate),
            alpha=alpha,
            precinct_labels=precinct_names,
            show_all_precincts=show_all_precincts,
            plot_as_histograms=plot_as_histograms,
            ax=ax,
        )
