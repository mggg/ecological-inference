"""
Models and fitting for 2x2 methods

TODO: Finish wakefield model
TODO: Truncated normal model
"""

import warnings
import pymc3 as pm
import numpy as np
from .plot_utils import (
    plot_conf_or_credible_interval,
    plot_boxplot,
    plot_kde,
    plot_precincts,
    plot_summary,
)

__all__ = ["TwoByTwoEI"]


def ei_beta_binom_model_modified(group_fraction, votes_fraction, precinct_pops, pareto_scale=8, pareto_shape=2):
    """
    An modification of the 2 x 2 beta/binomial EI model from King, Rosen, Tanner 1999,
    with (scaled) Pareto distributions over each parameters of the beta distribution,
    for better sampling geometry

    Parameters
    ----------
    group_fraction: Length-p (p=# of precincts) vector giving demographic information (X)
        as the fraction of precinct_pop in the demographic group of interest
    votes_fraction: Length p vector giving the fraction of each precinct_pop that votes
        for the candidate of interest (T)
    precinct_pops: Length-p vector giving size of each precinct population of interest
        (e.g. voting population) (N)

    Returns
    -------
    model: A pymc3 model

    Notes
    -----
    Reparametrizing of the hyperpriors to give (hopefully) better geometry for sampling.
    Also gives intuitive interpretation of hyperparams as mean and counts
    """
    votes_count_obs = votes_fraction * precinct_pops
    num_precincts = len(precinct_pops)
    with pm.Model() as model:
        phi_1 = pm.Uniform("phi_1", lower=0.0, upper=1.0)
        kappa_1 = pm.Pareto("kappa_1", m=pareto_scale, alpha=pareto_shape)

        phi_2 = pm.Uniform("phi_2", lower=0.0, upper=1.0)
        kappa_2 = pm.Pareto("kappa_2", m=pareto_scale, alpha=pareto_shape)

        b_1 = pm.Beta(
            "b_1",
            alpha=phi_1 * kappa_1,
            beta=(1.0 - phi_1) * kappa_1,
            shape=num_precincts,
        )
        b_2 = pm.Beta(
            "b_2",
            alpha=phi_2 * kappa_2,
            beta=(1.0 - phi_2) * kappa_2,
            shape=num_precincts,
        )

        theta = group_fraction * b_1 + (1 - group_fraction) * b_2
        pm.Binomial("votes_count", n=precinct_pops, p=theta, observed=votes_count_obs)
    return model


def ei_beta_binom_model(group_fraction, votes_fraction, precinct_pops, lmbda):
    """
    2 x 2 beta/binomial EI model from King, Rosen, Tanner 1999

    Parameters
    ----------
    group_fraction: Length-p (p=# of precincts) vector giving demographic information
        as the fraction of precinct_pop in the demographic group of interest
    votes_fraction: Length p vector giving the fraction of each precinct_pop that
        votes for the candidate of interest
    precinct_pops: Length-p vector giving size of each precinct population of interest
         (e.g. voting population)
    lmbda: a hyperparameter governing the exponential distributions
        over the parameters for the parameters of the beta distributions
        King sets this to 0.5, but lower seems to produce better geometry
        for sampling

    Returns
    -------
    model: A pymc3 model
    """

    votes_count_obs = votes_fraction * precinct_pops
    num_precincts = len(precinct_pops)
    with pm.Model() as model:
        c_1 = pm.Exponential("c_1", lmbda)
        d_1 = pm.Exponential("d_1", lmbda)
        c_2 = pm.Exponential("c_2", lmbda)
        d_2 = pm.Exponential("d_2", lmbda)

        b_1 = pm.Beta("b_1", alpha=c_1, beta=d_1, shape=num_precincts)
        b_2 = pm.Beta("b_2", alpha=c_2, beta=d_2, shape=num_precincts)

        theta = group_fraction * b_1 + (1 - group_fraction) * b_2
        pm.Binomial("votes_count", n=precinct_pops, p=theta, observed=votes_count_obs)
    return model


class TwoByTwoEI:
    """
    Fitting and plotting for king97, king99, and wakefield models
    """

    def __init__(self, model_name, **additional_model_params):
        # model_name can be 'king97', 'king99' or 'king99_pareto_modification' or 'wakefield'
        self.vote_fraction = None
        self.model_name = model_name
        self.additional_model_params = additional_model_params

        self.demographic_group_fraction = None
        self.votes_fraction = None
        self.precinct_pops = None
        self.precinct_names = None
        self.demographic_group_name = None
        self.candidate_name = None
        self.sim_model = None
        self.sim_trace = None
        self.sampled_voting_prefs = [None, None]
        self.posterior_mean_voting_prefs = [None, None]
        self.credible_interval_95_mean_voting_prefs = [None, None]

    def fit(
        self,
        group_fraction,
        votes_fraction,
        precinct_pops,
        demographic_group_name="given demographic group",
        candidate_name="given candidate",
        precinct_names=None,
    ):
        """Fit the specified model using MCMC sampling
        Required arguments:
        group_fraction  :   Length-p (p=# of precincts) vector giving demographic
                            information (X) as the fraction of precinct_pop in
                            the demographic group of interest
        votes_fraction  :   Length p vector giving the fraction of each precinct_pop
                            that votes for the candidate of interest (T)
        precinct_pops   :   Length-p vector giving size of each precinct population
                            of interest (e.g. voting population) (N)
        Optional arguments:
        demographic_group_name  :   Name of the demographic group of interest,
                                    where results are computed for the
                                    demographic group and its complement
        candidate_name          :   Name of the candidate whose support we
                                    want to analyze
        precinct_names          :   Length p vector giving the string names
                                    for each precinct.
        """
        # Additional params includes lambda for king99, the
        # parameter passed to the exponential hyperpriors,
        # and the paretoo_scale and pareto_shape parameters for the pareto
        # dist in the king99_pareto_modification model hyperprior
        self.demographic_group_fraction = group_fraction
        self.votes_fraction = votes_fraction
        self.precinct_pops = precinct_pops
        self.demographic_group_name = demographic_group_name
        self.candidate_name = candidate_name
        if precinct_names is not None:
            assert len(precinct_names) == len(precinct_pops)
            if len(set(precinct_names)) != len(precinct_names):
                warnings.warn(
                    "Precinct names are not unique. This may interfere with "
                    "passing precinct names to precinct_level_plot()."
                )
            self.precinct_names = precinct_names

        if self.model_name == "king99":
            self.sim_model = ei_beta_binom_model(
                group_fraction,
                votes_fraction,
                precinct_pops,
                **self.additional_model_params,
            )
        elif self.model_name == "king99_pareto_modification":
            self.sim_model = ei_beta_binom_model_modified(
                group_fraction, votes_fraction, precinct_pops, **self.additional_model_params
            )
        with self.sim_model:
            self.sim_trace = pm.sample(target_accept=0.99, tune=2000)

        self.calculate_summary()

    def calculate_summary(self):
        """Calculate point estimates (post. means) and credible intervals"""
        # multiply sample proportions by precinct pops to get samples of
        # number of voters the demographic group who voted for the candidate
        # in each precinct
        samples_converted_to_pops_gp1 = (
            self.sim_trace.get_values("b_1") * self.precinct_pops
        )  # shape: num_samples x num_precincts
        samples_converted_to_pops_gp2 = (
            self.sim_trace.get_values("b_2") * self.precinct_pops
        )  # shape: num_samples x num_precincts

        # obtain samples of total votes summed across all precinct for the candidate for each group
        samples_of_votes_summed_across_district_gp1 = samples_converted_to_pops_gp1.sum(axis=1)
        samples_of_votes_summed_across_district_gp2 = samples_converted_to_pops_gp2.sum(axis=1)

        # obtain samples of the districtwide proportion of each demog. group voting for candidate
        self.sampled_voting_prefs[0] = (
            samples_of_votes_summed_across_district_gp1 / self.precinct_pops.sum()
        )  # sampled voted prefs across precincts
        self.sampled_voting_prefs[1] = (
            samples_of_votes_summed_across_district_gp2 / self.precinct_pops.sum()
        )  # sampled voted prefs across precincts

        # compute point estimates
        self.posterior_mean_voting_prefs[0] = self.sampled_voting_prefs[0].mean()
        self.posterior_mean_voting_prefs[1] = self.sampled_voting_prefs[1].mean()

        # compute credible intervals
        percentiles = [2.5, 97.5]
        self.credible_interval_95_mean_voting_prefs[0] = np.percentile(
            self.sampled_voting_prefs[0], percentiles
        )
        self.credible_interval_95_mean_voting_prefs[1] = np.percentile(
            self.sampled_voting_prefs[1], percentiles
        )

    def summary(self):
        """Return a summary string"""
        # TODO: probably format this as a table
        return f"""Model: {self.model_name}
        Computed from the raw b_i samples by multiplying by population and then getting
        the proportion of the total pop (total pop=summed across all districts):
        The posterior mean for the district-level voting preference of
        {self.demographic_group_name} for {self.candidate_name} is
        {self.posterior_mean_voting_prefs[0]:.3f}
        The posterior mean for the district-level voting preference of
        non-{self.demographic_group_name} for {self.candidate_name} is
        {self.posterior_mean_voting_prefs[1]:.3f}
        95% Bayesian credible interval for district-level voting preference of
        {self.demographic_group_name} for {self.candidate_name} is
        {self.credible_interval_95_mean_voting_prefs[0]}
        95% Bayesian credible interval for district-level voting preference of
        non-{self.demographic_group_name} for {self.candidate_name} is
        {self.credible_interval_95_mean_voting_prefs[1]}
        """

    def precinct_level_estimates(self):
        """If desired, we can return precinct-level estimates"""

    def _voting_prefs(self):
        """Bundles together the samples, for ease of passing to plots"""
        return (
            self.sampled_voting_prefs[0],
            self.sampled_voting_prefs[1],
        )

    def _group_names_for_display(self):
        """Sets the group names to be displayed in plots"""
        return self.demographic_group_name, "non-" + self.demographic_group_name

    def plot_kde(self, ax=None):
        """kernel density estimate/ histogram plot"""
        return plot_kde(*self._voting_prefs(), *self._group_names_for_display(), ax=ax)

    def plot_boxplot(self, ax=None):
        """ Boxplot of voting prefs for each group"""
        return plot_boxplot(*self._voting_prefs(), *self._group_names_for_display(), ax=ax)

    def plot_intervals(self, ax=None):
        """ Plot of credible intervals for each group"""
        title = "95% credible intervals"
        return plot_conf_or_credible_interval(
            self.credible_interval_95_mean_voting_prefs[0],
            self.credible_interval_95_mean_voting_prefs[1],
            *self._group_names_for_display(),
            self.candidate_name,
            title,
            ax=ax,
        )

    def plot(self):
        """kde, boxplot, and credible intervals"""
        return plot_summary(
            *self._voting_prefs(),
            *self._group_names_for_display(),
            self.candidate_name,
        )

    def precinct_level_plot(self, ax=None, show_all_precincts=False, precinct_names=None):
        """Ridgeplots for precincts
        Optional arguments:
        ax                  :  matplotlib axes object
        show_all_precincts  :  If True, then it will show all ridge plots
                               (even if there are more than 50)
        precinct_names      :  Labels for each precinct (if not supplied, by
                               default we label each precinct with an integer
                               label, 1 to n)
        """
        voting_prefs_group1 = self.sim_trace.get_values("b_1")
        voting_prefs_group2 = self.sim_trace.get_values("b_2")
        if precinct_names is not None:
            precinct_idxs = [self.precinct_names.index(name) for name in precinct_names]
            voting_prefs_group1 = voting_prefs_group1[:, precinct_idxs]
            voting_prefs_group2 = voting_prefs_group2[:, precinct_idxs]
        return plot_precincts(
            voting_prefs_group1,
            voting_prefs_group2,
            precinct_labels=precinct_names,
            show_all_precincts=show_all_precincts,
            ax=ax,
        )
