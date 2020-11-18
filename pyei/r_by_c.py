"""
Models and fitting for rxc methods
where r and c are greater than or
equal to 2

TODO: Better or reparametrized priors for multinomial-dir
TODO: Greiner-Quinn Model
TODO: Refactor to integrate with two_by_two
"""


import warnings
import pymc3 as pm
import theano.tensor as tt
import numpy as np
from .plot_utils import plot_boxplots, plot_kdes, plot_intervals_all_precincts


def ei_multinom_dirichlet(group_fractions, votes_fractions, precinct_pops, lmbda=0.1):
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
        conc_params = pm.Exponential("conc_params", lam=lmbda, shape=(num_rows, num_cols))
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
        kappa = pm.Pareto("kappa", alpha=pareto_shape, m=pareto_scale, shape=num_rows)
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
    Fitting and plotting for multinomial-dirichlet and Greiner-Quinn EI models
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
        demographic_group_names,
        candidate_names,
        # precinct_names=None,
    ):
        """Fit the specified model using MCMC sampling
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

        # pylint: disable=duplicate-code
        # if precinct_names is not None:
        #     assert len(precinct_names) == len(precinct_pops)
        #     if len(set(precinct_names)) != len(precinct_names):
        #         warnings.warn(
        #             "Precinct names are not unique. This may interfere with "
        #             "passing precinct names to precinct_level_plot()."
        #         )
        #     self.precinct_names = precinct_names #TODO: set this
        # # pylint: enable=duplicate-code
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

        # TODO: Probably make the "modified" version the default option
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
        with self.sim_model:
            self.sim_trace = pm.sample(target_accept=0.99, tune=1001)

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

    def summary(self):
        """Return a summary string"""
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
                Credible interval:  {self.credible_interval_95_mean_voting_prefs[row][col]}
                """
                summary_str += summ
        return summary_str

    def precinct_level_estimates(self):
        """If desired, we can return precinct-level estimates

        Returns:
            precinct_posterior_means: num_precincts x r x c
            precinct_credible_intervals num_precincts x r x c x 2
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

    def candidate_of_choice(self):
        """ For each group, look at differences in preference within that group"""
        for row in range(self.num_groups_and_num_candidates[0]):
            print(self.demographic_group_names[row])
            for candidate_idx in range(self.num_groups_and_num_candidates[1]):
                frac = (
                    np.argmax(self.sampled_voting_prefs[:, row, :], axis=1) == candidate_idx
                ).sum() / self.sampled_voting_prefs.shape[0]

                print(
                    f"""In {frac} of samples, the district-level vote preference of
                    {self.demographic_group_names[row]} for {self.candidate_names[candidate_idx]}
                    was higher than for any other candidate"""
                )

    def plot_boxplots(self, plot_by="candidate", axes=None):
        """Plot boxplots of voting prefs (one boxplot for each candidate)

        plot_by: {'candidate', 'group'}
            If candidate, make one plot for each candidate. If group, make
            one subplot for each gropu
        axes: list of Matplotlib axis objects
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
        """ Kernel density plots of voting preference, plots grouped by candidate or group"""
        return plot_kdes(
            self.sampled_voting_prefs,
            self.demographic_group_names,
            self.candidate_names,
            plot_by=plot_by,
            axes=axes,
        )

    def plot_intervals_by_precinct(self, group_name, candidate_name):
        """ Plot of credible intervals for all precincts, for specified group and candidate"""
        if group_name not in self.demographic_group_names:
            raise ValueError(
                "group_name must be in the list of demographic_group_names provided to fit()"
            )

        if candidate_name not in self.candidate_names:
            raise ValueError(
                "candidate_name must be in the list of candidate_names provided to fit()"
            )

        group_index = self.demographic_group_names.index(group_name)
        candidate_index = self.candidate_names.index(candidate_name)

        point_estimates_all, intervals_all = self.precinct_level_estimates()
        point_estimates = point_estimates_all[:, group_index, candidate_index]
        intervals = intervals_all[:, group_index, candidate_index, :]

        plot_intervals_all_precincts(
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
                """Length of candiate_names should be equal to
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
