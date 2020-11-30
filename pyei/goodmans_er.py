"""
Goodman's ecological regression
"""

from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
import pymc3 as pm
from sklearn.linear_model import LinearRegression
from .plot_utils import (
    plot_conf_or_credible_interval,
    plot_boxplot,
    plot_kde,
    plot_summary
)

__all__ = ["GoodmansER"]


class GoodmansER:
    """
    Fitting and plotting for Goodman's ER (with options for pop weighting)
    """

    def __init__(self, is_weighted_regression=False):
        self.demographic_group_fraction = None
        self.vote_fraction = None
        self.demographic_group_fraction = None
        self.vote_fraction = None
        self.demographic_group_name = None
        self.candidate_name = None
        self.intercept_ = None
        self.slope_ = None
        self.voting_prefs_est_ = None
        self.voting_prefs_complement_est_ = None
        self.is_weighted_regression = is_weighted_regression

    def fit(
        self,
        group_fraction,
        vote_fraction,
        precinct_pops=None,
        demographic_group_name="given demographic group",
        candidate_name="given candidate",
    ):
        """Fit the linear model (use pop weights iff is_weighted_regression is true"""
        self.demographic_group_fraction = group_fraction
        self.vote_fraction = vote_fraction
        self.demographic_group_name = demographic_group_name
        self.candidate_name = candidate_name
        if self.is_weighted_regression:
            reg = LinearRegression().fit(
                group_fraction.reshape(-1, 1),
                vote_fraction,
                sample_weight=precinct_pops,
            )
        else:
            reg = LinearRegression().fit(group_fraction.reshape(-1, 1), vote_fraction)

        self.intercept_ = reg.intercept_
        self.slope_ = reg.coef_
        self.voting_prefs_est_ = reg.predict(np.array([1]).reshape(-1, 1))[0]
        self.voting_prefs_complement_est_ = reg.intercept_
        return self

    def summary(self):
        """Return summary of results as string"""
        if self.is_weighted_regression:
            model_name = "Goodmans ER, weighted by population"
        else:
            model_name = "Goodmans ER"
        return f"""{model_name}
        Est. fraction of {self.demographic_group_name}
        voters who voted for {self.candidate_name} is
        {self.voting_prefs_est_:.3f}
        Est. fraction of non- {self.demographic_group_name}
        voters who voted for {self.candidate_name} is
        {self.voting_prefs_complement_est_:.3f}
        """

    def plot(self):
        """Plot the linear regression with confidence interval"""
        fig, ax = plt.subplots()
        ax.axis("square")
        ax.grid(b=True, which="major")
        ax.set_ylim((0, 1))
        ax.set_xlim((0, 1))
        ax.set_xlabel(f"Fraction in group {self.demographic_group_name}")
        ax.set_ylabel(f"Fraction voting for {self.candidate_name}")
        sns.regplot(
            x=self.demographic_group_fraction,
            y=self.vote_fraction,
            ax=ax,
            ci=95,
            truncate=False,
        )
        return fig, ax


class GoodmansERBayes:
    def __init__(self, model_name, **additional_model_params):
        self.model_name = model_name
        self.sim_model = None
        self.sim_trace = None
        self.demographic_group_name = None
        self.additional_model_params = additional_model_params
        self.precinct_pops = None
        self.candidate_name = None

        self.posterior_mean_voting_prefs = [None, None]
        self.credible_interval_95_mean_voting_prefs = [None, None]
        self.sampled_voting_prefs = [None, None]
    
    def fit(self,
        group_fraction,
        vote_fraction,
        precinct_pops=None,
        demographic_group_name="given demographic group",
        candidate_name="given candidate"):

        self.precinct_pops = precinct_pops
        self.demographic_group_name = demographic_group_name
        self.candidate_name = candidate_name

        self.sim_model = goodmans_ER_bayes_model(group_fraction, vote_fraction, **self.additional_model_params)
        with self.sim_model:
            self.sim_trace = pm.sample(1000, tune=1000)

        self.calculate_summary()

    def _group_names_for_display(self):
        """Sets the group names to be displayed in plots"""
        return self.demographic_group_name, "non-" + self.demographic_group_name

    def calculate_summary(self):
        """Calculate point estimates (post. means) and credible intervals"""

        # obtain samples of the districtwide proportion of each demog. group voting for candidate
        self.sampled_voting_prefs[0] = (
            self.sim_trace.get_values("b_1")
        )  # sampled voted prefs across precincts
        self.sampled_voting_prefs[1] = (
            self.sim_trace.get_values("b_2")
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

    def _voting_prefs(self):
        """Bundles together the samples, for ease of passing to plots"""
        return (
            self.sampled_voting_prefs[0],
            self.sampled_voting_prefs[1],
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

    # def plot(self):
    # """Plot the linear regression with confidence interval"""

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
            [
                self.credible_interval_95_mean_voting_prefs[0],
                self.credible_interval_95_mean_voting_prefs[1],
            ],
            self._group_names_for_display(),
            self.candidate_name,
            title,
            ax=ax,
        )


def goodmans_ER_bayes_model(group_fraction, vote_fraction, sigma=1):
    with pm.Model() as bayes_er_model:
        b_1 = pm.Uniform("b_1")
        b_2 = pm.Uniform("b_2")

        eps = pm.HalfNormal('eps', sigma=sigma)

        pm.Normal("votes_count_obs", b_2 + (b_1 - b_2) * group_fraction, sigma=eps, observed=vote_fraction)
    
    return bayes_er_model