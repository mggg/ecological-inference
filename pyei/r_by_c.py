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
