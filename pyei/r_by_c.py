"""
Models and fitting for rxc methods
where r and c are greater than or
equal to 2

TODO: Fitting for multinomial-dir
TODO: Plotting for all r x c 
TODO: Greiner-Quinn Model
TODO: Refactor to integrate with two_by_two
TODO: error for model name that's not supported
"""


import warnings
import pymc3 as pm
import numpy as np
from .plot_utils import (
    plot_conf_or_credible_interval,
    plot_boxplot,
    plot_kdes,
    plot_precincts,
    plot_summary,
)


def ei_multinom_dirichlet(group_fractions, votes_fractions, precinct_pops):
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

    Returns
    -------
    model: A pymc3 model
    """

    num_precincts = len(precinct_pops)  # number of precincts
    r = group_fractions.shape[0]  # number of demographic groups
    c = votes_fractions.shape[0]  # number of candidates or voting outcomes

    # reshaping and rounding
    votes_count_obs = np.swapaxes(
        votes_fractions * precinct_pops, 0, 1
    ).round()  # num_precincts x r
    group_fractions_extended = np.expand_dims(group_fractions, axis=2)
    group_fractions_extended = np.repeat(group_fractions_extended, c, axis=2)
    group_fractions_extended = np.swapaxes(group_fractions_extended, 0, 1)  #  num_precincts x r x c

    with pm.Model() as model:
        # @TODO: are the prior conc_params what is in the literature? is it a good choice?
        conc_params = pm.Exponential("conc_params", lam=0.25, shape=(r, c))
        b = pm.Dirichlet("b", a=conc_params, shape=(num_precincts, r, c))  # num_precincts x r x c
        theta = (group_fractions_extended * b).sum(axis=1)
        pm.Multinomial(
            "votes_count", n=precinct_pops, p=theta, observed=votes_count_obs
        )  # num_precincts x r
    return model


class RowByColumnEI:
    """
    Fitting and plotting for multinomial-dirichlet and Greiner-Quinn EI models
    """

    def __init__(self, model_name, **additional_model_params):
        # model_name can be 'multinomial-dirichlet' or 'greiner-quinn'
        # TODO: implement greiner quinn
        self.demographic_group_fractions = None
        self.votes_fraction = None
        self.model_name = model_name
        self.additional_model_params = additional_model_params

        self.demographic_group_fractions = None
        self.votes_fractions = None
        self.precinct_pops = None
        self.precinct_names = None
        self.demographic_group_name = None
        self.candidate_name = None
        self.sim_trace = None
        self.sampled_voting_prefs = None
        self.posterior_mean_voting_prefs = None
        self.credible_interval_95_mean_voting_prefs = None

    def fit(
        self,
        group_fractions,
        votes_fractions,
        precinct_pops,
        demographic_group_names,
        candidate_names,
        precinct_names=None,
    ):
        """ Fit the specified model using MCMC sampling
            Required arguments:
            group_fractions :    r x p (p =#precincts = num_precicts) matrix giving demographic information 
                as the fraction of precinct_pop in the demographic group of interest for each of 
                p precincts and r demographic groups (sometimes denoted X)
            votes_fractions  :   c x p giving the fraction of each precinct_pop that votes
                for each of c candidates (sometimes denoted T)
            precinct_pops   :   Length-p vector giving size of each precinct population
                                of interest (e.g. voting population) (someteimes denoted N)
            Optional arguments:
            demographic_group_names  :   Names of the r demographic group of interest,
                                        where results are computed for the
                                        demographic group and its complement
            candidate_names          :   Name of the c candidates or voting outcomes of interest
            precinct_names          :   Length p vector giving the string names
                                        for each precinct.

        """
        # Additional params for hyperparameters
        # TODO: describe hyperparameters
        self.demographic_group_fractions = group_fractions
        self.votes_fractions = votes_fractions
        self.precinct_pops = precinct_pops
        self.demographic_group_names = demographic_group_names
        self.candidate_names = candidate_names
        if precinct_names is not None:
            assert len(precinct_names) == len(precinct_pops)
            if len(set(precinct_names)) != len(precinct_names):
                warnings.warn(
                    "Precinct names are not unique. This may interfere with "
                    "passing precinct names to precinct_level_plot()."
                )
            self.precinct_names = precinct_names
        self.num_groups_and_num_candidates = [
            group_fractions.shape[0],
            votes_fractions.shape[0],
        ]  # [r, c]

        # TODO: warning if num_groups from group_fractions doesn't matchu num_groups in demographic group_names

        if self.model_name == "multinomial-dirichlet":
            sim_model = ei_multinom_dirichlet(
                group_fractions, votes_fractions, precinct_pops, **self.additional_model_params,
            )
        with sim_model:
            self.sim_trace = pm.sample(target_accept=0.99, tune=1000)

        self.calculate_summary()

    def calculate_summary(self):
        """Calculate point estimates (post. means) and credible intervals"""
        # multiply sample proportions by precinct pops to get samples of
        # number of voters the demographic group who voted for the candidate
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
        for row in range(self.num_groups_and_num_candidates[0]):
            for col in range(self.num_groups_and_num_candidates[1]):
                self.credible_interval_95_mean_voting_prefs[row][col] = np.percentile(
                    self.sampled_voting_prefs[:, row, col], percentiles
                )

    def summary(self):
        """Return a summary string"""
        # TODO: probably format this as a table
        summary_str = """
            Computed from the raw b_ samples by multiplying by population and then getting
                the proportion of the total pop (total pop=summed across all districts):
            """
        for row in range(self.num_groups_and_num_candidates[0]):
            for col in range(self.num_groups_and_num_candidates[1]):
                s = f"""The posterior mean for the district-level voting preference of
                {self.demographic_group_names[row]} for {self.candidate_names[col]} is
                {self.posterior_mean_voting_prefs[row][col]:.3f}
                Credible interval:  {self.credible_interval_95_mean_voting_prefs[row][col]}
                """
                summary_str += s
        return summary_str
