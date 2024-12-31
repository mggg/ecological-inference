"""
Goodman's ecological regression
"""

from matplotlib import pyplot as plt
import numpy as np
import pymc as pm
from sklearn.linear_model import LinearRegression
import seaborn as sns
import seaborn.algorithms as sns_algo
import seaborn.utils as sns_utils

from .two_by_two import TwoByTwoEIBaseBayes

__all__ = ["GoodmansER", "GoodmansERBayes"]


class GoodmansER:
    """
    Fitting and plotting for Goodman's ER (with options for pop weighting)
    """

    def __init__(self, is_weighted_regression=False):
        """
        Parameters
        ----------
        is_weighted_regression: bool, optional
            Default is False. If true, weight precincts by population when
            performing the regression.
        """
        self.demographic_group_fraction = None
        self.vote_fraction = None
        self.demographic_group_fraction = None
        self.demographic_group_name = None
        self.candidate_name = None
        self.intercept_ = None
        self.slope_ = None
        self.voting_prefs_est_ = None
        self.voting_prefs_complement_est_ = None
        self.precinct_pops = None
        self.is_weighted_regression = is_weighted_regression

    def fit(
        self,
        group_fraction,
        vote_fraction,
        precinct_pops=None,
        demographic_group_name="given demographic group",
        candidate_name="given candidate",
    ):
        """Fit the linear model (use pop weights iff is_weighted_regression is true

        Parameters
        ----------
        group_fraction  :   Length-p (p=# of precincts) vector giving demographic
                            information (X) as the fraction of precinct_pop in
                            the demographic group of interest
        vote_fraction  :   Length p vector giving the fraction of each precinct_pop
                            that votes for the candidate of interest (T)
        precinct_pops   :   Length-p vector giving size of each precinct population
                            of interest (e.g. voting population) (N)
                            Used for weighting the regression if the GoodmansER
                            object has is_weighted_regression True
        demographic_group_name  :   Name of the demographic group of interest,
                                    where results are computed for the
                                    demographic group and its complement
        candidate_name          :   Name of the candidate whose support we
                                    want to analyze
        """
        self.demographic_group_fraction = group_fraction
        self.vote_fraction = vote_fraction
        self.precinct_pops = precinct_pops
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
        self.slope_ = reg.coef_[0]
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

    def plot(
        self,
        line_kws=None,
        scatter_kws=None,
        **sns_regplot_args,
    ):
        """Plot the linear regression with 95% confidence interval

        Notes:
        ------
        Can pass additional plot arguments through to seaborn.regplot, e.g.
        to change the line color:
            line_kws=dict(color="red")
            scatter_kws={"s": 50}
        """
        fig, ax = plt.subplots()
        ax.axis("square")
        ax.grid(visible=True, which="major")
        ax.set_ylim((0, 1))
        ax.set_xlim((0, 1))
        ax.set_xlabel(f"Fraction in group {self.demographic_group_name}")
        ax.set_ylabel(f"Fraction voting for {self.candidate_name}")

        if self.is_weighted_regression:
            if line_kws is None:
                line_kws = {}
            line_kws.setdefault("linewidth", 2.5)
            if scatter_kws is None:
                scatter_kws = {}
            scatter_kws.setdefault("s", 50)
            scatter_kws.setdefault("linewidths", 0)
            scatter_kws.setdefault("alpha", 0.8)
            sns.lineplot(
                x=[0, 1], y=[self.intercept_, self.intercept_ + self.slope_], ax=ax, **line_kws
            )
            sns.scatterplot(
                x=self.demographic_group_fraction, y=self.vote_fraction, ax=ax, **scatter_kws
            )
            xgrid = np.linspace(0, 1, 101)

            def fit_fast(xgrid, x, y, w):
                """
                Modifying this function from sns to accomodate weighted regression
                in CI computation
                """

                def weighted_reg_func(_x, _y, _w):
                    """Low-level regression and prediction using linear algebra."""
                    _w_sqrt = np.sqrt(_w)
                    x_weighted = np.diag(_w_sqrt).dot(_x)
                    y_weighted = np.diag(_w_sqrt).dot(_y)
                    return np.linalg.pinv(x_weighted).dot(y_weighted)

                x_with_ones = np.c_[np.ones(len(x)), x]
                grid = np.c_[np.ones(len(xgrid)), xgrid]  # append ones for intercept
                yhat = grid.dot(weighted_reg_func(x_with_ones, y, w))
                beta_boots = sns_algo.bootstrap(
                    x_with_ones, y, w, func=weighted_reg_func, n_boot=1000, units=None, seed=None
                ).T
                yhat_boots = grid.dot(beta_boots).T
                return yhat, yhat_boots

            _, yhat_boots = fit_fast(
                xgrid,
                self.demographic_group_fraction,
                self.vote_fraction,
                self.precinct_pops,
            )
            # CIs at each grid point
            err_bands = sns_utils.ci(yhat_boots, which=95, axis=0)
            ax.fill_between(
                xgrid, *err_bands, alpha=0.15, color=plt.gca().lines[-1].get_color()
            )  # match color of plot
        else:
            sns.regplot(
                x=self.demographic_group_fraction,
                y=self.vote_fraction,
                ax=ax,
                ci=95,
                truncate=False,
                **sns_regplot_args,
            )
        return fig, ax


class GoodmansERBayes(TwoByTwoEIBaseBayes):
    """Bayesian ecological regression with uniform prior over the voting preferences
    Generate samples from the posterior.
    """

    def __init__(
        self,
        model_name="goodman_er_bayes",
        weighted_by_pop=False,
        **additional_model_params,
    ):
        """
        Optional arguments:
        model_name: str
            Default is "goodman_er_bayes"
        weighted_by_pop: bool
            Default is False. If true, weight precincts by population when
            performing the regression.
        additional_model_parameters:
            Any hyperparameters for model
        """
        # TODO if no other model name is applicable here, remove need for model_name argument
        super().__init__(model_name, **additional_model_params)
        self.weighted_by_pop = weighted_by_pop

    def fit(
        self,
        group_fraction,
        votes_fraction,
        precinct_pops=None,
        demographic_group_name="given demographic group",
        candidate_name="given candidate",
        target_accept=0.9,
        tune=1000,
        **other_sampling_args,
    ):
        """Fit a bayesian er modeling via sampling.

        Parameters
        ----------
        group_fraction  :   Length-p (p=# of precincts) vector giving demographic
                            information (X) as the fraction of precinct_pop in
                            the demographic group of interest
        votes_fraction  :   Length p vector giving the fraction of each precinct_pop
                            that votes for the candidate of interest (T)
        precinct_pops   :   Length-p vector giving size of each precinct population
                            of interest (e.g. voting population) (N)
        demographic_group_name  :   Name of the demographic group of interest,
                                    where results are computed for the
                                    demographic group and its complement
        candidate_name          :   Name of the candidate whose support we
                                    want to analyze
        precinct_names          :   Length p vector giving the string names
                                    for each precinct.
        target_accept : float, optional
            Default=.99 Strictly between zero and 1 (should be close to 1). Passed to pymc's
            sampling.sample
        tune : int, optional
            Default=1500, passed to pymc's sampling.sample
        other_sampling_args :
            For to pymc's sampling.sample
            https://docs.pymc.io/api/inference.html
        """
        self.precinct_pops = precinct_pops
        self.demographic_group_name = demographic_group_name
        self.candidate_name = candidate_name
        self.demographic_group_fraction = group_fraction
        self.votes_fraction = votes_fraction

        if self.weighted_by_pop:
            model_function = goodmans_er_bayes_pop_weighted_model
            self.additional_model_params["precinct_pops"] = precinct_pops
        else:
            model_function = goodmans_er_bayes_model
        self.sim_model = model_function(
            group_fraction, votes_fraction, **self.additional_model_params
        )

        with self.sim_model:  # pylint: disable=not-context-manager
            self.sim_trace = pm.sample(
                1000, tune=tune, target_accept=target_accept, **other_sampling_args
            )

        self.calculate_sampled_voting_prefs()
        super().calculate_summary()

    def calculate_sampled_voting_prefs(self):
        """Sets sampled_voting_prefs"""
        # obtain samples of the districtwide proportion of each demog. group voting for candidate
        self.sampled_voting_prefs[0] = (
            self.sim_trace["posterior"]["b_1"].stack(all_draws=["chain", "draw"]).values.T
        )
        # sampled voted prefs across precincts
        self.sampled_voting_prefs[1] = (
            self.sim_trace["posterior"]["b_2"].stack(all_draws=["chain", "draw"]).values.T
        )
        # sampled voted prefs across precincts

    def compute_credible_int_for_line(self, x_vals=np.linspace(0, 1, 100)):
        """Computes regression line (mean) and 95% central credible interval for
        the mean line at each of the specified x values(x_vals)

        Parameters
        ----------
        x_vals : numpy array, optional
            Default: np.linspace(0, 1, 100). 1-dimensional numpy array of values between
            0 and 1, for which the function should compute the mean line and 95%
            credible interval. Each element of x_vals represents the fraction of the population
            that is in the demographic group of interest (values for X in the notation of King '97)

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

    def plot(self, scatter_kws=None, line_kws=None):
        """Plot regression line of votes_fraction vs. group_fraction, with scatter plot and
        equal-tailed 95% credible interval for the line"
        Parameters:
        -----------
        scatter_kws : dict (None)
            Keyword arguments to be passed to matplotlib.Axes.scatter
        line_kws : dict (None)
            Keyword arguments to be passed to matplotlib.Axes.plot.
            Note that the color of the ccredible interval shading is set to match
            the color of the line itself (but the shading has higher alpha)

        Notes:
        ------
        Examples of additional kwargs. Scatter_colors is list of colors of length num_precincts
        scatter_kws={"c": scatter_colors, "color": None, "s": 20},
        line_kws={"color":"black", "lw": 1}
        """
        # TODO: consider renaming these plots for goodman, to disambiguate with TwoByTwoEI.plot()
        # TODO: accept axis argument
        x_vals, means, lower_bounds, upper_bounds = self.compute_credible_int_for_line()
        _, ax = plt.subplots()

        if scatter_kws is None:
            scatter_kws = {}
        if line_kws is None:
            line_kws = {}
        scatter_kws.setdefault("color", "steelblue")
        scatter_kws.setdefault("alpha", 0.8)
        line_kws.setdefault("color", "steelblue")

        ax.axis("square")
        ax.set_xlabel(f"Fraction in group {self.demographic_group_name}")
        ax.set_ylabel(f"Fraction voting for {self.candidate_name}")
        ax.scatter(
            self.demographic_group_fraction,
            self.votes_fraction,
            **scatter_kws,
        )
        ax.plot(x_vals, means, **line_kws)
        ax.fill_between(x_vals, lower_bounds, upper_bounds, alpha=0.2, color=line_kws["color"])
        ax.grid()
        ax.set_xlim((0, 1))
        ax.set_ylim((0, 1))
        return ax


def goodmans_er_bayes_model(group_fraction, votes_fraction, sigma=1):
    """Ecological regression with uniform priors over voting prefs b_1, b_2,
    constraining them to be between zero and 1

    Parameters
    ----------
    group_fraction  :   Length-p (p=# of precincts) vector giving demographic
                            information (X) as the fraction of precinct_pop in
                            the demographic group of interest
    votes_fraction  :   Length p vector giving the fraction of each precinct_pop
                            that votes for the candidate of interest (T)

    Returns
    -------
    bayes_er_model: a pymc3 model
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

    Parameters
    ----------
    group_fraction  :   Length-p (p=# of precincts) vector giving demographic
                            information (X) as the fraction of precinct_pop in
                            the demographic group of interest
    votes_fraction  :   Length p vector giving the fraction of each precinct_pop
                            that votes for the candidate of interest (T)
    precinct_pops   :   Length-p vector giving size of each precinct population
                            of interest (e.g. voting population) (N)

    Returns
    -------
    bayes_er_model: a pymc3 model
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
