"""
Models and fitting for 2x2 methods

TODO: Checks for wakefield model
TODO: Wakefield model with normal prior
TODO: Truncated normal model
"""

import warnings
import pymc3 as pm
import numpy as np
import theano.tensor as tt
import theano
from .plot_utils import (
    plot_conf_or_credible_interval,
    plot_boxplot,
    plot_kde,
    plot_precincts,
    plot_polarization_kde,
    plot_summary,
    plot_intervals_all_precincts,
)

__all__ = ["TwoByTwoEI", "ei_beta_binom_model_modified"]


def ei_beta_binom_model_modified(
    group_fraction, votes_fraction, precinct_pops, pareto_scale=8, pareto_shape=2
):
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
    pareto_scale: A positive real number. The scale paremeter passed to the
        pareto hyperparamters
    pareto_shape: A positive real number. The shape paremeter passed to the
        pareto hyperparamters

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
    # tot_pop = precinct_pops.sum()
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
        # pm.Deterministic("voting_prefs_gp1", (b_1 * precinct_pops).sum() / tot_pop)
        # pm.Deterministic("voting_prefs_gp2", (b_2 * precinct_pops).sum() / tot_pop)

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


def log_binom_sum(lower, upper, obs_vote, n0_curr, n1_curr, b_1_curr, b_2_curr, prev):
    """
    Helper function for computing log prob of convolution of binomial
    """

    # votes_within_group_count is y_0i in Wakefield's notation, the count of votes from
    # given group for given candidate within precinct i (unobserved)
    votes_within_group_count = tt.arange(lower, upper)
    component_for_current_precinct = pm.math.logsumexp(
        pm.Binomial.dist(n0_curr, b_1_curr).logp(votes_within_group_count)
        + pm.Binomial.dist(n1_curr, b_2_curr).logp(obs_vote - votes_within_group_count)
    )
    return prev + component_for_current_precinct


def binom_conv_log_p(b_1, b_2, n_0, n_1, upper, lower, obs_votes):
    """
    Log probability for convolution of binomials

    Parameters
    ----------
    b_1: corresponds to p0 in wakefield's notation, the probability that an individual
        in the given demographic group votes for the given candidate
    b_2: corresponds to p1 in wakefield's notation, the probability that an individual
        in the complement of the given demographic group votes for the given candidate

    n_0: the count of given demographic group in the precinct
    n_1: the count of the complement of given demographic group in the precinct

    lower, upper : lower and upper bounds on the (unobserved) count of votes from given
    deographic group for given candidate within precinct
    (corresponds to votes_within_group_count in log_binom_sum and y_0i in Wakefield's)

    Returns
    -------
    A theano tensor giving the log probability of b_1, b_2 (given the other parameters)

    Notes
    -----
    See Wakefield 2004 equation 4
    """

    result, _ = theano.scan(
        fn=log_binom_sum,
        outputs_info={"taps": [-1], "initial": tt.as_tensor(np.array([0.0]))},
        sequences=[
            tt.as_tensor(lower),
            tt.as_tensor(upper),
            tt.as_tensor(obs_votes),
            tt.as_tensor(n_0),
            tt.as_tensor(n_1),
            tt.as_tensor(b_1),
            tt.as_tensor(b_2),
        ],
    )
    return result[-1]


def wakefield_model_beta(
    group_fraction, votes_fraction, precinct_pops, pareto_scale=8, pareto_shape=2
):
    """
    2 x 2 EI model Wakefield

    """

    vote_count_obs = votes_fraction * precinct_pops
    group_count_obs = group_fraction * precinct_pops
    num_precincts = len(precinct_pops)
    upper = np.minimum(group_count_obs, vote_count_obs)  # upper bound on y
    lower = np.maximum(0.0, vote_count_obs - precinct_pops + group_count_obs)  # lower bound on y
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

        pm.DensityDist(
            "votes_count_obs",
            binom_conv_log_p,
            observed={
                "b_1": b_1,
                "b_2": b_2,
                "n_0": group_count_obs,
                "n_1": precinct_pops - group_count_obs,
                "upper": upper,
                "lower": lower,
                "obs_votes": vote_count_obs,
            },
        )
    return model


def wakefield_normal(group_fraction, votes_fraction, precinct_pops, mu0=0, mu1=0):
    """
    2 x 2 EI model Wakefield with normal hyperpriors

    Note: Wakefield suggests adding another level of hierarchy, with a prior over mu0 and mu1,
    sigma0, sigma1, but that is not yet implemented here

    """

    vote_count_obs = votes_fraction * precinct_pops
    group_count_obs = group_fraction * precinct_pops
    num_precincts = len(precinct_pops)
    upper = np.minimum(group_count_obs, vote_count_obs)  # upper bound on y
    lower = np.maximum(0.0, vote_count_obs - precinct_pops + group_count_obs)  # lower bound on y
    with pm.Model() as model:
        sigma_0 = pm.Gamma("sigma0", 1, 0.1)
        sigma_1 = pm.Gamma("sigma1", 1, 0.1)

        theta_0 = pm.Normal("theta0", mu0, sigma_0, shape=num_precincts)
        theta_1 = pm.Normal("theta1", mu1, sigma_1, shape=num_precincts)

        b_1 = pm.Deterministic(
            "b_1", tt.exp(theta_0) / (1 + tt.exp(theta_0))
        )  # vector of length num_precincts
        b_2 = pm.Deterministic(
            "b_2", tt.exp(theta_1) / (1 + tt.exp(theta_1))
        )  # vector of length num_precincts

        pm.DensityDist(
            "votes_count_obs",
            binom_conv_log_p,
            observed={
                "b_1": b_1,
                "b_2": b_2,
                "n_0": group_count_obs,
                "n_1": precinct_pops - group_count_obs,
                "upper": upper,
                "lower": lower,
                "obs_votes": vote_count_obs,
            },
        )
    return model


class TwoByTwoEIBaseBayes:
    """
    Init, summary, plots for 2 x 2 EI models that proceed via sampling

    Note: does not assume precinct level samples are available,
    because this is not possible for goodman_er_bayes. So, plots of precinct
    level quanitities are defined in the subclass TwoByTwoEi

    Note: subclass will need to define methods to fit (sample), and
        to define summary quantities
    """

    def __init__(self, model_name, **additional_model_params):
        self.model_name = model_name
        self.additional_model_params = additional_model_params

        self.sim_model = None
        self.sim_trace = None

        self.precinct_pops = None
        self.demographic_group_name = None
        self.candidate_name = None

        self.demographic_group_fraction = None
        self.votes_fraction = None

        self.posterior_mean_voting_prefs = [None, None]
        self.credible_interval_95_mean_voting_prefs = [None, None]
        self.sampled_voting_prefs = [None, None]

    def _group_names_for_display(self):
        """Sets the group names to be displayed in plots"""
        return self.demographic_group_name, "non-" + self.demographic_group_name

    def _voting_prefs(self):
        """Bundles together the samples, for ease of passing to plots"""
        return (self.sampled_voting_prefs[0], self.sampled_voting_prefs[1])

    def calculate_summary(self):
        """Calculate point estimates (post. means) and credible intervals
        Assumes sampled_voting_prefs has already been set"""

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

    def _calculate_polarization(self, threshold=None, probability=None, reference_group=0):
        """calculate percentile given a threshold, or threshold if given a probability
        exactly one of probabilty and threshold must be null
        """
        samples = self.sampled_voting_prefs[1] - self.sampled_voting_prefs[0]
        group = self.demographic_group_name
        group_complement = "non-" + self.demographic_group_name
        if reference_group == 1:
            samples = -samples
            group = "non-" + self.demographic_group_name
            group_complement = self.demographic_group_name

        if probability is None and threshold is not None:
            probability = (samples > threshold).sum() / len(self.sampled_voting_prefs[0])
        elif threshold is None and probability is not None:
            threshold = np.percentile(samples, probability)
        else:
            raise ValueError(
                """Exactly one of threshold or probability must be None.
            Set a threshold to calculate the associated probability, or a probability
            to calculate the associated probability/percentile
            """
            )
        return threshold, probability, samples, group, group_complement

    def polarization_report(
        self, threshold=None, probability=None, reference_group=0, verbose=True
    ):
        """return probabiity that the difference between the group's
        preferences for the given candidate is more than threshold
        """
        threshold, probability, _, group, group_complement = self._calculate_polarization(
            threshold, probability, reference_group
        )
        if verbose:
            return f"""The probability that the difference between the groups' preferences
            for {self.candidate_name} ( {group_complement} - {group} ) iss more than
            {threshold:.5f} is {probability:.5f}"""
        else:
            return probability

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

    def plot_polarization_kde(
        self, threshold=None, probability=None, reference_group=0, show_threshold=False, ax=None
    ):
        """Plot kde of differences between voting preferences"""
        threshold, probability, samples, group, group_complement = self._calculate_polarization(
            threshold, probability, reference_group
        )
        return plot_polarization_kde(
            samples,
            threshold,
            probability,
            group,
            group_complement,
            self.candidate_name,
            show_threshold,
            ax,
        )


class TwoByTwoEI(TwoByTwoEIBaseBayes):
    """
    Fitting and plotting for king97, king99, and wakefield models
    """

    def __init__(self, model_name, **additional_model_params):
        # model_name can be 'king97', 'king99' or 'king99_pareto_modification'
        # 'wakefield_beta' or 'wakefield normal'
        super().__init__(model_name, **additional_model_params)

        self.precinct_pops = None
        self.precinct_names = None

    def fit(
        self,
        group_fraction,
        votes_fraction,
        precinct_pops,
        demographic_group_name="given demographic group",
        candidate_name="given candidate",
        precinct_names=None,
        target_accept=0.99,
        tune=1500,
        draw_samples=True,
        **other_sampling_args,
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
        target_accept : float, optional
            Default=.99 Strictly between zero and 1 (should be close to 1). Passed to pymc's
            sampling.sample
        tune : int, optional
            Default=1500, passed to pymc's sampling.sample
        draw_samples: bool, optional
            Default=True. Set to False to only set up the variable but not generate
            posterior samples (i.e. if you want to generate prior predictive samples only)
        other_sampling_args :
            For to pymc's sampling.sample
            https://docs.pymc.io/api/inference.html
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
            model_function = ei_beta_binom_model
            # self.sim_model = ei_beta_binom_model(
            #     group_fraction,
            #     votes_fraction,
            #     precinct_pops,
            #     **self.additional_model_params,
        elif self.model_name == "king99_pareto_modification":
            model_function = ei_beta_binom_model_modified
            # self.sim_model = ei_beta_binom_model_modified(
            #     group_fraction,
            #     votes_fraction,
            #     precinct_pops,
            #     **self.additional_model_params,
            # )
        elif self.model_name == "wakefield_beta":
            model_function = wakefield_model_beta
            # self.sim_model = wakefield_model_beta(
            #     group_fraction,
            #     votes_fraction,
            #     precinct_pops,
            #     **self.additional_model_params,
            # )
        elif self.model_name == "wakefield_normal":
            model_function = wakefield_normal
            # self.sim_model = wakefield_normal(
            #     group_fraction,
            #     votes_fraction,
            #     precinct_pops,
            #     **self.additional_model_params,
            # )

        self.sim_model = model_function(
            group_fraction,
            votes_fraction,
            precinct_pops,
            **self.additional_model_params,
        )

        if draw_samples:
            # TODO: this workaround shouldn't be necessary. Modify the model so that the checks
            # can run without error
            if self.model_name == "wakefield_beta" or self.model_name == "wakefield_normal":
                compute_convergence_checks = False
                print("WARNING: some convergence checks currently disabled for wakefield model")
            else:
                compute_convergence_checks = True

            with self.sim_model:
                self.sim_trace = pm.sample(
                    target_accept=target_accept,
                    tune=tune,
                    compute_convergence_checks=compute_convergence_checks,
                    **other_sampling_args,
                )

            self.calculate_sampled_voting_prefs()
            super().calculate_summary()

    def calculate_sampled_voting_prefs(self):
        """Sampled voting preferences (combining samples with precinct pops)"""
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

    def precinct_level_estimates(self):
        """If desired, we can return precinct-level estimates"""
        # TODO: make this output match r_by_c version in shape,
        percentiles = [2.5, 97.5]
        precinct_level_samples_gp1 = self.sim_trace.get_values("b_1")
        precinct_posterior_means_gp1 = precinct_level_samples_gp1.mean(axis=0)
        precinct_credible_intervals_gp1 = np.percentile(
            precinct_level_samples_gp1, percentiles, axis=0
        ).T

        precinct_level_samples_gp2 = self.sim_trace.get_values("b_2")
        precinct_posterior_means_gp2 = precinct_level_samples_gp2.mean(axis=0)
        precinct_credible_intervals_gp2 = np.percentile(
            precinct_level_samples_gp2, percentiles, axis=0
        ).T

        return (
            precinct_posterior_means_gp1,
            precinct_posterior_means_gp2,
            precinct_credible_intervals_gp1,
            precinct_credible_intervals_gp2,
        )

    def plot_intervals_by_precinct(self):
        """ Plot of pointe estimates and credible intervals for each precinct"""
        # TODO: Fix use of axes
        (
            precinct_posterior_means_gp1,
            precinct_posterior_means_gp2,
            precinct_credible_intervals_gp1,
            precinct_credible_intervals_gp2,
        ) = self.precinct_level_estimates()

        plot_gp1 = plot_intervals_all_precincts(
            precinct_posterior_means_gp1,
            precinct_credible_intervals_gp1,
            self.candidate_name,
            self.precinct_names,
            self._group_names_for_display()[0],
        )
        plot_gp2 = plot_intervals_all_precincts(
            precinct_posterior_means_gp2,
            precinct_credible_intervals_gp2,
            self.candidate_name,
            self.precinct_names,
            self._group_names_for_display()[1],
        )

        return plot_gp1, plot_gp2

    def plot(self, axes=None):
        """kde, boxplot, and credible intervals"""
        return plot_summary(
            *self._voting_prefs(),
            *self._group_names_for_display(),
            self.candidate_name,
            axes=axes,
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
