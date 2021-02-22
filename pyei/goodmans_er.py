"""
Goodman's ecological regression
"""

from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
import pymc3 as pm
from sklearn.linear_model import LinearRegression

from .two_by_two import TwoByTwoEIBaseBayes

__all__ = ["GoodmansER"]


class GoodmansER:
    """
    Fitting and plotting for Goodman's ER (with options for pop weighting)
    """

    def __init__(self, is_weighted_regression=False):
        self.demographic_group_fraction = None
        self.vote_fraction = None
        self.demographic_group_fraction = None
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


class GoodmansERBayes(TwoByTwoEIBaseBayes):
    """Bayesian ecological regression with uniform prior over the voting preferences
    Generate samples from the posterior.
    """

    def __init__(self, model_name, weighted_by_pop=False, **additional_model_params):
        super().__init__(model_name, **additional_model_params)
        self.weighted_by_pop = weighted_by_pop

    def fit(
        self,
        group_fraction,
        votes_fraction,
        precinct_pops=None,
        demographic_group_name="given demographic group",
        candidate_name="given candidate",
    ):
        """Fit a bayesian er modeling via sampling."""
        self.precinct_pops = precinct_pops
        self.demographic_group_name = demographic_group_name
        self.candidate_name = candidate_name
        self.demographic_group_fraction = group_fraction
        self.votes_fraction = votes_fraction

        if self.weighted_by_pop:
            model_function = goodmans_er_bayes_pop_weighted_model
        else:
            model_function = goodmans_er_bayes_model
        self.sim_model = model_function(
            group_fraction, votes_fraction, precinct_pops, **self.additional_model_params
        )

        with self.sim_model:
            self.sim_trace = pm.sample(1000, tune=1000, target_accept=0.9)

        self.calculate_sampled_voting_prefs()
        super().calculate_summary()

    def calculate_sampled_voting_prefs(self):
        """Sets sampled_voting_prefs"""
        # obtain samples of the districtwide proportion of each demog. group voting for candidate
        self.sampled_voting_prefs[0] = self.sim_trace.get_values(
            "b_1"
        )  # sampled voted prefs across precincts
        self.sampled_voting_prefs[1] = self.sim_trace.get_values(
            "b_2"
        )  # sampled voted prefs across precincts

    def compute_credible_int_for_line(self, x_vals=np.linspace(0, 1, 100)):
        """Computes regression line (mean) and 95% credible interval for the
        mean line at each of the specified x values(x_vals)
        """
        lower_bounds = np.empty_like(x_vals)
        upper_bounds = np.empty_like(x_vals)
        means = np.empty_like(x_vals)
        for idx, x in enumerate(x_vals):
            mean_samples = (
                self.sampled_voting_prefs[1]
                + (self.sampled_voting_prefs[0] - self.sampled_voting_prefs[1]) * x
            )
            percentiles = np.percentile(mean_samples, [2.5, 97.5])
            lower_bounds[idx] = percentiles[0]
            upper_bounds[idx] = percentiles[1]
            means[idx] = mean_samples.mean()

        return x_vals, means, lower_bounds, upper_bounds

    def plot(self):
        """Plot regression line of votes_fraction vs. group_fraction, with scatter plot and
        95% credible interval for the line"""
        # TODO: consider renaming these plots for goodman, to disambiguate with TwoByTwoEI.plot()
        # TODO: accept axis argument
        x_vals, means, lower_bounds, upper_bounds = self.compute_credible_int_for_line()
        _, ax = plt.subplots()
        ax.axis("square")
        ax.set_xlabel(f"Fraction in group {self.demographic_group_name}")
        ax.set_ylabel(f"Fraction voting for {self.candidate_name}")
        ax.scatter(self.demographic_group_fraction, self.votes_fraction, alpha=0.8)
        ax.plot(x_vals, means)
        ax.fill_between(x_vals, lower_bounds, upper_bounds, color="steelblue", alpha=0.2)
        ax.grid()
        ax.set_xlim((0, 1))
        ax.set_ylim((0, 1))


def goodmans_er_bayes_model(group_fraction, votes_fraction, sigma=1):
    """Ecological regression with uniform priors over voting prefs b_1, b_2,
    constraining them to be between zero and 1
    """
    with pm.Model() as bayes_er_model:
        b_1 = pm.Uniform("b_1")
        b_2 = pm.Uniform("b_2")

        eps = pm.HalfNormal("eps", sigma=sigma)

        pm.Normal(
            "votes_count_obs",
            b_2 + (b_1 - b_2) * group_fraction,
            sigma=eps,
            observed=votes_fraction,
        )

    return bayes_er_model


def goodmans_er_bayes_pop_weighted_model(group_fraction, votes_fraction, precinct_pops, sigma=1):
    """Ecological regression with variance of modeled vote fraction inversely proportional to
    precinct population.

    Uniform priors over voting prefs b_1, b_2 constrain them to be between 0 and 1
    """

    mean_precinct_pop = precinct_pops.mean()
    with pm.Model() as bayes_er_model:
        b_1 = pm.Uniform("b_1")
        b_2 = pm.Uniform("b_2")

        eps = pm.HalfNormal("eps", sigma=sigma)

        pm.Normal(
            "votes_count_obs",
            b_2 + (b_1 - b_2) * group_fraction,
            sigma=eps * pm.math.sqrt(mean_precinct_pop / precinct_pops),
            observed=votes_fraction,
        )

    return bayes_er_model
