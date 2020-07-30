"""
Goodman's ecological regression
"""

import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression


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
            self.demographic_group_fraction,
            self.vote_fraction,
            ax=ax,
            ci=95,
            truncate=False,
        )
        return fig, ax
