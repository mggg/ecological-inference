"""
Models and fitting for 2x2 methods

"""

import warnings
import pymc as pm
from pymc import sampling_jax
import numpy as np
import aesara.tensor as at
import aesara
from .plot_utils import (
    plot_conf_or_credible_interval,
    plot_boxplots,
    plot_kdes,
    plot_precincts,
    plot_polarization_kde,
    plot_summary,
    plot_intervals_all_precincts,
)

__all__ = ["TwoByTwoEI", "ei_beta_binom_model_modified"]


def truncated_normal_asym(
    group_fraction, votes_fraction, precinct_pops
):  # pylint: disable=too-many-locals
    """
    A modification of king97's truncated normal that puts some broad priors
    over the parameters of the truncated normal dist

    Parameters
    ----------
    group_fraction: Length-p (p=# of precincts) vector giving demographic information
        as the fraction of precinct_pop in the demographic group of interest
    votes_fraction: Length p vector giving the fraction of each precinct_pop that
        votes for the candidate of interest
    precinct_pops: Length-p vector giving size of each precinct population of interest
         (e.g. voting population)

    Returns
    -------
    model: A pymc3 model

    """
    num_precincts = len(precinct_pops)
    b_1_l_bound = np.maximum(0, (votes_fraction - 1 + group_fraction) / group_fraction)
    b_1_u_bound = np.minimum(1, votes_fraction / group_fraction)
    b_2_l_bound = np.maximum(0, (votes_fraction - group_fraction) / (1 - group_fraction))
    b_2_u_bound = np.minimum(1, (votes_fraction) / (1 - group_fraction))

    # For stability, use whichever of b_1 and b_2 has the broader width between bounds as
    # the parameter that we include in the model (then the other of b_1 or b_2 will
    # be calculated deterministically from that and votes_fraction) -- the deterministically
    # calculated one is here marked as "lower-level" one
    if (b_2_u_bound - b_2_l_bound).mean() > (b_1_u_bound - b_1_l_bound).mean():  # swap
        group_fraction = 1 - group_fraction
        upper_level_b_name = "b_2"
        lower_level_b_name = "b_1"
        upper_level_tn_mean_name = "tn_mean_2"
        lower_level_tn_mean_name = "tn_mean_1"
        upper_level_sigma_name = "sigma_22"
        lower_level_sigma_name = "sigma_11"
        upper_level_l_bound = b_2_l_bound
        upper_level_u_bound = b_2_u_bound
    else:  # not swapping
        upper_level_b_name = "b_1"
        lower_level_b_name = "b_2"
        upper_level_tn_mean_name = "tn_mean_1"
        lower_level_tn_mean_name = "tn_mean_2"
        upper_level_sigma_name = "sigma_11"
        lower_level_sigma_name = "sigma_22"
        upper_level_l_bound = b_1_l_bound
        upper_level_u_bound = b_1_u_bound

    with pm.Model() as model:
        sigma_upper = pm.HalfNormal(
            upper_level_sigma_name, sigma=0.707
        )  # chosen to match King 97 #sigma11
        sigma_lower = pm.HalfNormal(
            lower_level_sigma_name, sigma=0.707
        )  # chosen to match king 97 #sigma22
        rho = pm.Uniform("rho", -0.5, 0.5)  # TODO: revisit
        sigma_12 = sigma_upper * sigma_lower * rho

        tn_mean_upper = pm.Uniform(upper_level_tn_mean_name)
        tn_mean_lower = pm.Uniform(lower_level_tn_mean_name)

        upper_b = pm.TruncatedNormal(
            upper_level_b_name,
            mu=tn_mean_upper,
            sigma=sigma_upper,
            shape=num_precincts,
            lower=upper_level_l_bound,
            upper=upper_level_u_bound,
        )

        mu_i = tn_mean_upper * group_fraction + tn_mean_lower * (1 - group_fraction)
        w_i = pm.Deterministic(
            "w_i", sigma_upper**2 * group_fraction + sigma_12 * (1 - group_fraction)
        )
        sigma_i_sq = pm.Deterministic(
            "sigma_i_sq",
            sigma_lower**2
            + 2 * (sigma_12 - sigma_lower**2) * group_fraction
            + (sigma_upper * 2 + sigma_lower**2 - 2 * sigma_12) * group_fraction**2,
        )

        votes_frac_mean = mu_i + w_i * (upper_b - tn_mean_upper) / (sigma_upper**2)
        votes_frac_var = pm.Deterministic(
            "votes_frac_var", sigma_i_sq - w_i**2 / (sigma_upper**2)
        )

        votes_frac_l_bound = group_fraction * upper_b
        votes_frac_u_bound = (1 - group_fraction) + group_fraction * upper_b

        votes_frac_stdev = pm.math.sqrt(votes_frac_var)
        pm.TruncatedNormal(
            "votes_fraction",
            mu=votes_frac_mean,
            sigma=votes_frac_stdev,
            lower=votes_frac_l_bound,
            upper=votes_frac_u_bound,
            observed=votes_fraction,
        )
        pm.Deterministic(
            lower_level_b_name, (votes_fraction - upper_b * group_fraction) / (1 - group_fraction)
        )
    return model


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

    Parameters
    ----------
    lower, upper : lower and upper bounds on the (unobserved) count of votes from given
        deographic group for given candidate within precinct
    n0_curr: the (current value for the) count of given demographic group in the precinct
    n1_curr: the (current value for the)count of the complement of given demographic group
         in the precinct
    b_1_curr: corresponds to p0 in wakefield's notation, the probability that an individual
        in the given demographic group votes for the given candidate
    b_2_curr: corresponds to p1 in wakefield's notation, the probability that an individual
        in the complement of the given demographic group votes for the given candidate
    prev: the partial sum of the log prob of the convolution of binomials

    Returns
    -------
    prev + component_for_current_precinct : sum of the previous value of the partial sum
        for the log prob of the convolution of binomials and an additional term for one
        precinct

    """

    # votes_within_group_count is y_0i in Wakefield's notation, the count of votes from
    # given group for given candidate within precinct i (unobserved)
    votes_within_group_count = at.arange(lower, upper)
    component_for_current_precinct = pm.math.logsumexp(
        pm.logp(pm.Binomial.dist(n0_curr, b_1_curr), votes_within_group_count)
        + pm.logp(pm.Binomial.dist(n1_curr, b_2_curr), obs_vote - votes_within_group_count)
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

    result, _ = aesara.scan(
        fn=log_binom_sum,
        outputs_info={"taps": [-1], "initial": at.as_tensor(np.array([0.0]))},
        sequences=[
            at.as_tensor(lower),
            at.as_tensor(upper),
            at.as_tensor(obs_votes),
            at.as_tensor(n_0),
            at.as_tensor(n_1),
            at.as_tensor(b_1),
            at.as_tensor(b_2),
        ],
    )
    return result[-1]


# def wakefield_model_beta(
#     group_fraction, votes_fraction, precinct_pops, pareto_scale=8, pareto_shape=2
# ):
#     """
#     2 x 2 EI model based on Wakefield's, with pareto distributions in upper level of hierarchy

#     Parameters
#     ----------
#     group_fraction: Length-p (p=# of precincts) vector giving demographic information
#         as the fraction of precinct_pop in the demographic group of interest
#     votes_fraction: Length p vector giving the fraction of each precinct_pop that
#         votes for the candidate of interest
#     precinct_pops: Length-p vector giving size of each precinct population of interest
#          (e.g. voting population)

#     Returns
#     -------
#     model: A pymc3 model
#     """

#     vote_count_obs = votes_fraction * precinct_pops
#     group_count_obs = group_fraction * precinct_pops
#     num_precincts = len(precinct_pops)
#     upper = np.minimum(group_count_obs, vote_count_obs)  # upper bound on y
#     lower = np.maximum(0.0, vote_count_obs - precinct_pops + group_count_obs)  # lower bound on y
#     with pm.Model() as model:
#         phi_1 = pm.Uniform("phi_1", lower=0.0, upper=1.0)
#         kappa_1 = pm.Pareto("kappa_1", m=pareto_scale, alpha=pareto_shape)

#         phi_2 = pm.Uniform("phi_2", lower=0.0, upper=1.0)
#         kappa_2 = pm.Pareto("kappa_2", m=pareto_scale, alpha=pareto_shape)

#         b_1 = pm.Beta(
#             "b_1",
#             alpha=phi_1 * kappa_1,
#             beta=(1.0 - phi_1) * kappa_1,
#             shape=num_precincts,
#         )
#         b_2 = pm.Beta(
#             "b_2",
#             alpha=phi_2 * kappa_2,
#             beta=(1.0 - phi_2) * kappa_2,
#             shape=num_precincts,
#         )

#         pm.DensityDist(
#             "votes_count_obs",
#             b_1,
#             b_2,
#             group_count_obs,
#             precinct_pops - group_count_obs,
#             upper,
#             lower,
#             observed=vote_count_obs,
#             logp=binom_conv_log_p,
#         )
#     return model


# def wakefield_normal(group_fraction, votes_fraction, precinct_pops, mu0=0, mu1=0):
#     """
#     2 x 2 EI model Wakefield with normal hyperpriors

#     Note: Wakefield suggests adding another level of hierarchy, with a prior over mu0 and mu1,
#     sigma0, sigma1, but that is not yet implemented here

#     Parameters
#     ----------
#     group_fraction: Length-p (p=# of precincts) vector giving demographic information
#         as the fraction of precinct_pop in the demographic group of interest
#     votes_fraction: Length p vector giving the fraction of each precinct_pop that
#         votes for the candidate of interest
#     precinct_pops: Length-p vector giving size of each precinct population of interest
#          (e.g. voting population)
#     mu0: float
#         Mean of the normally distributed hyperparameter associated with the demographic
#         group of interest
#     m1: Mean of the normally distributed hyperparameter associated with the complement
#     of the demographic group of interest

#     Returns
#     -------
#     model: A pymc3 model

#     """

#     vote_count_obs = votes_fraction * precinct_pops
#     group_count_obs = group_fraction * precinct_pops
#     num_precincts = len(precinct_pops)
#     upper = np.minimum(group_count_obs, vote_count_obs)  # upper bound on y
#     lower = np.maximum(0.0, vote_count_obs - precinct_pops + group_count_obs)  # lower bound on y
#     with pm.Model() as model:
#         sigma_0 = pm.Gamma("sigma0", 1, 0.1)
#         sigma_1 = pm.Gamma("sigma1", 1, 0.1)

#         theta_0 = pm.Normal("theta0", mu0, sigma_0, shape=num_precincts)
#         theta_1 = pm.Normal("theta1", mu1, sigma_1, shape=num_precincts)

#         b_1 = pm.Deterministic(
#             "b_1", at.exp(theta_0) / (1 + at.exp(theta_0))
#         )  # vector of length num_precincts
#         b_2 = pm.Deterministic(
#             "b_2", at.exp(theta_1) / (1 + at.exp(theta_1))
#         )  # vector of length num_precincts

#         pm.DensityDist(
#             "votes_count_obs",
#             b_1,
#             b_2,
#             group_count_obs,  # n_0
#             precinct_pops - group_count_obs,  # n_1
#             upper,
#             lower,
#             observed=vote_count_obs,  # obs_votes
#             logp=binom_conv_log_p,
#         )
#     return model


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
        """
        model_name: str
            The name of one of the models ( "king99", "king99_pareto_modification",
             "truncated_normal",
            "goodman_er_bayes")
        additional_model_params
            Hyperparameters to pass to model, if changing default parameters
            (see model documentation for the hyperparameters for each model)
        """
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

    def group_names_for_display(self):
        """Returns the group names to be displayed in plots"""
        return self.demographic_group_name, "non-" + self.demographic_group_name

    def _voting_prefs_array(self):
        """Bundles together the samples as num_samples x 2 x 1 array,
        for ease of passing to plots"""
        num_samples = len(self.sampled_voting_prefs[0])
        sampled_voting_prefs = np.empty((num_samples, 2, 1))  # num_samples x 2 x 1
        sampled_voting_prefs[:, 0, 0] = self.sampled_voting_prefs[0]
        sampled_voting_prefs[:, 1, 0] = self.sampled_voting_prefs[1]
        return sampled_voting_prefs

    def calculate_summary(self):
        """Calculate point estimates (post. means) and 95% equal-tailed credible intervals
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

    def _calculate_polarization(self, threshold=None, percentile=None, reference_group=0):
        """calculate percentile given a threshold, or threshold if given a percentile
        exactly one of percentile and threshold must be null

        Parameters
        ----------
        threshold: float
            If not None, function will return the estimated probability that difference
            between the two groups' preferences for the given candidate is more than
            {threshold}
        percentile: float
            If not None, function will return the threshold for which {percentile}
            equals the estimated probability that difference between the two groups'
            preferences for the given candidate is more than {threshold}
        reference group: int {0, 1}
            The index of the reference group. If 0, the thresholds are calcuated as
            (group 0 preferences - group 1 preferences). If 1, the thresholds are
            calculated as (group 1 preferences - group 0 preferences)

        Notes
        -----
        Exactly one of threshold and percentile must be None

        """
        samples = self.sampled_voting_prefs[0] - self.sampled_voting_prefs[1]
        group = self.demographic_group_name
        group_complement = "non-" + self.demographic_group_name
        if reference_group == 1:
            samples = -samples
            group = "non-" + self.demographic_group_name
            group_complement = self.demographic_group_name

        if percentile is None and threshold is not None:
            percentile = 100 * (samples > threshold).sum() / len(self.sampled_voting_prefs[0])
        elif threshold is None and percentile is not None:
            threshold = np.percentile(samples, 100 - percentile)
        else:
            raise ValueError(
                """Exactly one of threshold or percentile must be None.
            Set a threshold to calculate the associated percentile, or a percentile
            to calculate the associated threshold.
            """
            )
        return threshold, percentile, samples, [group, group_complement]

    def polarization_report(self, threshold=None, percentile=None, reference_group=0, verbose=True):
        """
        For a given threshold, return the probability that difference between the group's
        preferences for the given candidate is more than threshold
        OR
        For a given confidence level, return the associated central credible interval for
        the difference between the two groups' preferences.
        Exactly one {percentile,threshold} must be None

        Parameters
        ----------
        threshold: float
            If not None, function will return the estimated probability that difference
            between the two groups' preferences for the given candidate is more than
            {threshold}
        percentile: float
            If not None, function will return the central interval which {percentile}
            equals the estimated probability that difference between the two groups'
            preferences for the given candidate is in that interval
        reference group: int {0, 1}
            The index of the reference group. If 0, the thresholds are calcuated as
            (group 0 preferences - group 1 preferences). If 1, the thresholds are
            calculated as (group 1 preferences - group 0 preferences)
        verbose: bool
            If True, print out a report

        Notes
        -----
        Exactly one of threshold and percentile must be None
        """
        return_interval = threshold is None

        if return_interval:
            lower_percentile = (100 - percentile) / 2
            upper_percentile = lower_percentile + percentile
            lower_threshold, _, _, groups = self._calculate_polarization(
                threshold, upper_percentile, reference_group
            )
            upper_threshold, _, _, groups = self._calculate_polarization(
                threshold, lower_percentile, reference_group
            )
            if verbose:
                print(
                    f"There is a {percentile}% probability that the difference between the groups'"
                    + f" preferences for {self.candidate_name} ({groups[0]} - {groups[1]}) is "
                    + f"between [{lower_threshold:.2f}, {upper_threshold:.2f}]."
                )
            return (lower_threshold, upper_threshold)
        else:
            threshold, percentile, _, groups = self._calculate_polarization(
                threshold, percentile, reference_group
            )
            if verbose:
                print(
                    f"There is a {percentile:.1f}% probability that the difference between"
                    + f" the groups' preferences for {self.candidate_name} ({groups[0]} -"
                    + f"  {groups[1]}) is more than {threshold:.2f}."
                )
            return percentile

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
        95% equal-tailed Bayesian credible interval for district-level voting preference of
        {self.demographic_group_name} for {self.candidate_name} is
        {self.credible_interval_95_mean_voting_prefs[0]}
        95% equal-tailed Bayesian credible interval for district-level voting preference of
        non-{self.demographic_group_name} for {self.candidate_name} is
        {self.credible_interval_95_mean_voting_prefs[1]}
        """

    def plot_kde(self, ax=None):
        """kernel density estimate/ histogram plot
        Optional arguments:
        ax  :  matplotlib axes object
        """
        return plot_kdes(
            self._voting_prefs_array(),
            self.group_names_for_display(),
            [self.candidate_name],
            plot_by="candidate",
            axes=ax,
        )

    def plot_boxplot(self, ax=None):
        """Boxplot of voting prefs for each group
        Optional arguments:
        ax  :  matplotlib axes object"""
        return plot_boxplots(
            self._voting_prefs_array(),
            self.group_names_for_display(),
            [self.candidate_name],
            plot_by="candidate",
            axes=ax,
        )

    def plot_intervals(self, ax=None):
        """Plot of credible intervals for each group
        Optional arguments:
        ax  :  matplotlib axes object
        """
        title = "95% credible intervals"
        return plot_conf_or_credible_interval(
            [
                self.credible_interval_95_mean_voting_prefs[0],
                self.credible_interval_95_mean_voting_prefs[1],
            ],
            self.group_names_for_display(),
            self.candidate_name,
            title,
            ax=ax,
        )

    def plot_polarization_kde(
        self, threshold=None, percentile=None, reference_group=0, show_threshold=False, ax=None
    ):
        """
        Plot kde of differences between voting preferences

        Parameters
        ----------
        threshold: float
            If not None, function will return the estimated probability that difference
            between the two groups' preferences for the given candidate is more than
            {threshold}
        percentile: float
            If not None, function will return the central interval which {percentile}
            equals the estimated probability that difference between the two groups'
            preferences for the given candidate is in that interval
        reference group: int {0, 1}
            The index of the reference group. If 0, the thresholds are calcuated as
            (group 0 preferences - group 1 preferences). If 1, the thresholds are
            calculated as (group 1 preferences - group 0 preferences)
        show_threshold: bool (optional)
            Default: False. If true, add a vertical line at the threshold on the plot
            and display the associated tail probability
        ax: matplotlib axis object (optional)

        Returns
        -------
        Matplotlib axis object

        """
        return_interval = threshold is None

        if return_interval:
            lower_percentile = (100 - percentile) / 2
            upper_percentile = lower_percentile + percentile
            lower_threshold, _, samples, groups = self._calculate_polarization(
                threshold, upper_percentile, reference_group
            )
            upper_threshold, _, samples, groups = self._calculate_polarization(
                threshold, lower_percentile, reference_group
            )
            thresholds = [lower_threshold, upper_threshold]
        else:
            threshold, percentile, samples, groups = self._calculate_polarization(
                threshold, percentile, reference_group
            )
            thresholds = [threshold]

        return plot_polarization_kde(
            samples,
            thresholds,
            percentile,
            groups,
            self.candidate_name,
            show_threshold,
            ax,
        )


class TwoByTwoEI(TwoByTwoEIBaseBayes):
    """
    Fitting and plotting for king97, king99, and wakefield models
    """

    def __init__(self, model_name, **additional_model_params):
        """
        model_name: str
            Name of model: can be 'king97', 'king99', 'king99_pareto_modification'
            'wakefield_beta' or 'wakefield normal'
        additional_model_params
            Hyperparameters to pass to model, if changing default parameters
            (see model documentation for the hyperparameters for each model)
        """
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

        # check that lengths of group_fraction, votes_fraction, precinct_pops match
        if not (
            len(group_fraction) == len(votes_fraction) and len(votes_fraction) == len(precinct_pops)
        ):
            raise ValueError(
                """Mismatching num_precincts in inputs. \n
            votes_fraction, group_fraction, precinct_pops should all have same length
            """
            )
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
            self.precinct_names = np.array(precinct_names)

        if self.model_name == "king99":
            model_function = ei_beta_binom_model

        elif self.model_name == "king99_pareto_modification":
            model_function = ei_beta_binom_model_modified

        elif self.model_name == "truncated_normal":
            model_function = truncated_normal_asym

        self.sim_model = model_function(
            group_fraction,
            votes_fraction,
            precinct_pops,
            **self.additional_model_params,
        )

        if draw_samples:
            with self.sim_model:  # pylint: disable=not-context-manager
                # this "if" is a workaround until jax.scipy.special.erfcx is
                # implemented https://github.com/google/jax/issues/1987
                # (when that's implemented, trunc-normal can use jax sampling
                # as well) @TODO: check on this in a little while
                if self.model_name in [
                    "truncated_normal"
                ]:  # , "wakefield_normal", "wakefield_beta"]:
                    self.sim_trace = pm.sample(
                        target_accept=target_accept,
                        tune=tune,
                        **other_sampling_args,
                    )
                else:
                    self.sim_trace = sampling_jax.sample_numpyro_nuts(
                        target_accept=target_accept,
                        tune=tune,
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
            self.sim_trace["posterior"]["b_1"].stack(all_draws=["chain", "draw"]).values.T
            * self.precinct_pops
        )  # shape: num_samples x num_precincts
        samples_converted_to_pops_gp2 = (
            self.sim_trace["posterior"]["b_2"].stack(all_draws=["chain", "draw"]).values.T
            * self.precinct_pops
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
        """If desired, we can return precinct-level estimates
        Returns:
            precinct_posterior_means: num_precincts x 2 (groups) x 2 (candidates)
            precinct_credible_intervals: num_precincts x 2 (groups) x 2 (candidates) x 2 (endpoints)
        """
        # TODO: make this output match r_by_c version in shape, num_precincts x 2 x 2
        percentiles = [2.5, 97.5]
        num_precincts = len(self.precinct_pops)

        # The stracking on the next line convers to a num_samples x num_precincts array
        precinct_level_samples_gp1 = (
            self.sim_trace["posterior"]["b_1"].stack(all_draws=["chain", "draw"]).values.T
        )
        precinct_posterior_means_gp1 = precinct_level_samples_gp1.mean(axis=0)
        precinct_credible_intervals_gp1 = np.percentile(
            precinct_level_samples_gp1, percentiles, axis=0
        ).T

        # The stracking on the next line convers to a num_samples x num_precincts array
        precinct_level_samples_gp2 = (
            self.sim_trace["posterior"]["b_2"].stack(all_draws=["chain", "draw"]).values.T
        )
        precinct_posterior_means_gp2 = precinct_level_samples_gp2.mean(axis=0)
        precinct_credible_intervals_gp2 = np.percentile(
            precinct_level_samples_gp2, percentiles, axis=0
        ).T  # num_precincts x 2

        precinct_posterior_means = np.empty((num_precincts, 2, 2))
        precinct_posterior_means[:, 0, 0] = precinct_posterior_means_gp1
        precinct_posterior_means[:, 0, 1] = 1 - precinct_posterior_means_gp1
        precinct_posterior_means[:, 1, 0] = precinct_posterior_means_gp2
        precinct_posterior_means[:, 1, 1] = 1 - precinct_posterior_means_gp2

        precinct_credible_intervals = np.empty((num_precincts, 2, 2, 2))
        precinct_credible_intervals[:, 0, 0, :] = precinct_credible_intervals_gp1
        precinct_credible_intervals[:, 0, 1, :] = 1 - precinct_credible_intervals_gp1
        precinct_credible_intervals[:, 1, 0, :] = precinct_credible_intervals_gp2
        precinct_credible_intervals[:, 1, 1, :] = 1 - precinct_credible_intervals_gp2

        return (precinct_posterior_means, precinct_credible_intervals)

    def plot_intervals_by_precinct(self):
        """Plot of point estimates and credible intervals for each precinct"""
        # TODO: Fix use of axes

        precinct_posterior_means, precinct_credible_intervals = self.precinct_level_estimates()
        precinct_posterior_means_gp1 = precinct_posterior_means[:, 0, 0]
        precinct_posterior_means_gp2 = precinct_posterior_means[:, 1, 0]
        precinct_credible_intervals_gp1 = precinct_credible_intervals[:, 0, 0, :]
        precinct_credible_intervals_gp2 = precinct_credible_intervals[:, 1, 0, :]

        plot_gp1 = plot_intervals_all_precincts(
            precinct_posterior_means_gp1,
            precinct_credible_intervals_gp1,
            self.candidate_name,
            self.precinct_names,
            self.group_names_for_display()[0],
        )
        plot_gp2 = plot_intervals_all_precincts(
            precinct_posterior_means_gp2,
            precinct_credible_intervals_gp2,
            self.candidate_name,
            self.precinct_names,
            self.group_names_for_display()[1],
        )

        return plot_gp1, plot_gp2

    def plot(self, axes=None):
        """kde, boxplot, and credible intervals
        Optional arguments:
        axes : list or tuple of matplotlib axis objects or None
            Default=None
            Length 2: (ax_box, ax_hist)
        """
        return plot_summary(
            self._voting_prefs_array(),
            self.group_names_for_display()[0],
            self.group_names_for_display()[1],
            self.candidate_name,
            axes=axes,
        )

    def precinct_level_plot(
        self,
        ax=None,
        alpha=1,
        show_all_precincts=False,
        precinct_names=None,
        plot_as_histograms=False,
    ):
        """Ridgeplots for precincts
        Optional arguments:
        ax                  :  matplotlib axes object
        show_all_precincts  :  If True, then it will show all ridge plots
                               (even if there are more than 50)
        alpha               : float
                                The opacity for the fill color in the
                                kdes / histograms
        precinct_names      :  Labels for each precinct (if not supplied, by
                               default we label each precinct with an integer
                               label, 1 to n)
        plot_as_histograms : bool, optional. Default is false. If true, plot
                                with histograms instead of kdes
        """
        voting_prefs_group1 = (
            self.sim_trace["posterior"]["b_1"].stack(all_draws=["chain", "draw"]).values.T
        )
        voting_prefs_group2 = (
            self.sim_trace["posterior"]["b_2"].stack(all_draws=["chain", "draw"]).values.T
        )
        group_names = self.group_names_for_display()
        if precinct_names is not None:
            precinct_idxs = np.arange(len(self.precinct_names))
            voting_prefs_group1 = voting_prefs_group1[:, precinct_idxs]
            voting_prefs_group2 = voting_prefs_group2[:, precinct_idxs]
        return plot_precincts(
            [voting_prefs_group1, voting_prefs_group2],
            group_names=group_names,
            candidate=self.candidate_name,
            alpha=alpha,
            precinct_labels=precinct_names,
            show_all_precincts=show_all_precincts,
            plot_as_histograms=plot_as_histograms,
            ax=ax,
        )
