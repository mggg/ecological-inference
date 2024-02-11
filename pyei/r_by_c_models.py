"""
Functions that return pymc models for RowByColumnEI
"""

import pymc as pm
import pytensor.tensor as at
import numpy as np

__all__ = ["ei_multinom_dirichlet_modified", "ei_multinom_dirichlet"]


def ei_multinom_dirichlet(group_fractions, votes_fractions, precinct_pops, lmbda1=4, lmbda2=2):
    """
    An implementation of the r x c dirichlet/multinomial EI model

    Parameters:
    -----------
    group_fractions: r x num_precincts  matrix giving demographic information
        as the fraction of precinct_pop in the demographic group of interest for each of
        p precincts and r demographic groups (sometimes denoted X)
    votes_fractions: c x num_precincts matrix giving the fraction of each precinct_pop that votes
        for each of c candidates (sometimes denoted T)
    precinct_pops: Length-num_precincts vector giving size of each precinct population of interest
        (e.g. voting population) (sometimes denoted N)
    lmbda1: float parameter passed to the Gamma(lmbda, 1/lmbda2) distribution
    lmbda2: float parameter passed to the Gamma(lmbda, 1/lmbda2) distribution

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
        # conc_params = pm.Exponential("conc_params", lam=lmbda, shape=(num_rows, num_cols))
        conc_params = pm.Gamma(
            "conc_params", alpha=lmbda1, beta=1 / lmbda2, shape=(num_rows, num_cols)
        )  # chosen to match eiPack
        beta = pm.Dirichlet("b", a=conc_params, shape=(num_precincts, num_rows, num_cols))
        # num_precincts x r x c
        theta = (group_fractions_extended * beta).sum(axis=1)
        pm.Multinomial(
            "votes_count",
            n=precinct_pops,
            p=theta,
            observed=votes_count_obs,
            shape=(num_precincts, num_cols),
        )  # num_precincts x c
    return model


def ei_multinom_dirichlet_modified(
    group_fractions, votes_fractions, precinct_pops, pareto_scale=5, pareto_shape=1
):
    """
    An implementation of the r x c dirichlet/multinomial EI model with reparametrized hyperpriors

    Parameters:
    -----------
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
        kappa = pm.Pareto("kappa", alpha=pareto_shape, m=pareto_scale, shape=num_rows)  # size r
        phi = pm.Dirichlet("phi", a=np.ones(num_cols), shape=(num_rows, num_cols))  # r x c
        phi_kappa = pm.Deterministic("phi_kappa", at.transpose(kappa * at.transpose(phi)))
        beta = pm.Dirichlet("b", a=phi_kappa, shape=(num_precincts, num_rows, num_cols))
        # num_precincts x r x c
        theta = (group_fractions_extended * beta).sum(axis=1)  # sum across num_rows
        pm.Multinomial(
            "votes_count",
            n=precinct_pops,
            p=theta,
            observed=votes_count_obs,
            shape=(num_precincts, num_cols),
        )  # num_precincts x c
    return model
