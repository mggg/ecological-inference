"""
Utilities for RowByColumnEI
"""

import warnings


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
                """Length of candidate_names should be equal to
            c = votes_fractions.shape[0]. If not, plotting labels be inaccurate.
            """
            )

    print(f"Running {demographic_group_names} x {candidate_names} EI")
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
    # check shapes of group_fractions, votes_fractions, and precinct_pops to make sure
    # number of precincts match (the last dimension of each)
    if not (
        len(votes_fractions[0]) == len(group_fractions[0])
        and len(group_fractions[0]) == len(precinct_pops)
    ):
        raise ValueError(
            """Mismatching num_precincts in input shapes. Inputs should have shape: \n
        votes_fraction shape: r x num_precincts \n
        group_fractions shape: c x num_precincts \n
        precinct_pops length: num_precincts
        """
        )
