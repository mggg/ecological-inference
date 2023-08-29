"""
Functionality for Gibbs sampler to generate posterior samples
from model from James Greiner, D. and Quinn, K.M., 2009.
R× C ecological inference: bounds, correlations, flexibility
and transparency of assumptions. Journal of the Royal Statistical
Society: Series A (Statistics in Society), 172(1), pp.67-81.
"""

import random
import scipy.stats as st
import numpy as np
import arviz as az
from tqdm import tqdm
from numba import njit, prange
from pyei.distribution_utils import non_central_hypergeometric_sample

__all__ = ["pyei_greiner_quinn_sample", "greiner_quinn_gibbs_sample"]


def pyei_greiner_quinn_sample(  # pylint: disable=too-many-locals
    group_fractions,
    votes_fractions,
    precinct_pops,
    num_samples=None,
    burnin=None,
    nu_0=None,
    psi_0=None,
    k_0_inv=None,
    mu_0=None,
    gamma=0.1,
):
    """
    Converts pyei inputs to counts for gq, sets default hyperparams,
    runs sampler, returns samples as an arviz InferenceData object

    Parameters:
    -----------
    group_fractions :   r x p (p =#precincts = num_precincts) matrix giving demographic
            information as the fraction of precinct_pop in the demographic group for each
            of p precincts and r demographic groups (sometimes denoted X)
    votes_fractions  :  c x p giving the fraction of each precinct_pop that votes
        for each of c candidates (sometimes denoted T)
    precinct_pops   :   Length-p array of ints giving size of each precinct population
                        of interest (e.g. voting population) (someteimes denoted N)
    num_samples: int (optional)
        number of MCMC samples to draw using the Gibbs sampler
    nu_0: int (optional)
        hyperparameter (df) scalar
        roughly interpretable as number of pseudodistricts for prior
    psi_0: ndarray (optional)
        hyperparameter (scale) - square matrix r * (c - 1) x r * (c - 1)
    k_0_inv: (optional)
        hyperparameter - square matrix r * (c - 1) x r * (c - 1)
        Precision for prior distribution over mu, the mean of the distribution of omega
        mu | mu_0, k_0 ~ N(mu_0, k_0) = N(mu_0, k_0_inv^{-1})
        omega | mu, Sigma ~ N(mu, Sigma)
    mu_0: ndarray (optional)
        vector of length r * (c - 1)
        Governs prior distribution over mu, the mean of the distribution of omega
        mu | mu_0, k_0 ~ N(mu_0, k_0)
        omega | mu, Sigma ~ N(mu, Sigma)
    gamma: float (optional)
        parameter of proposal dist in the Metropolis step for sampling theta
    burnin: int
        the number of samples at the start of the MCMC chain to be discarded

    Returns:
    --------
    idata: arviz.InverenceData
        An InferenceData object containing the posterior samples
        obtained via the Gibbs sampler

    Notes:
    -----
    This function converts group_fraction and votes_fraction to counts
    using the precinct_pops in the following manner: first,
    the fractions are multiplied by the precinct population and each rounded
    if the sum of the counts obtained by rounding don't match the precinct
    populations, the difference is subtracted from a *random* group or column
    """
    if num_samples is None:
        num_samples = 500_000
    if burnin is None:
        burnin = 100_000

    # covert fractions to counts
    # and make counts match precinct_pops (for example)
    # if a mismatch, add or subtract from a random group/candidate
    num_groups = group_fractions.shape[0]
    num_candidates = votes_fractions.shape[0]

    group_counts = np.round(group_fractions * precinct_pops)
    vote_counts = np.round(votes_fractions * precinct_pops)

    group_diff = group_counts.sum(axis=0) - precinct_pops
    for idx_of_mismatch in np.where(group_diff != 0):
        group_to_adjust = random.randint(0, num_groups - 1)
        group_counts[group_to_adjust, idx_of_mismatch] -= group_diff[idx_of_mismatch]

    vote_diff = vote_counts.sum(axis=0) - precinct_pops
    for idx_of_mismatch in np.where(vote_diff != 0):
        candidate_to_adjust = random.randint(0, num_candidates - 1)
        vote_counts[candidate_to_adjust, idx_of_mismatch] -= vote_diff[idx_of_mismatch]

    group_counts = group_counts.T
    vote_counts = vote_counts.T

    # Set any unspecified hyperparameters to default values
    if nu_0 is None:
        nu_0 = 10
    if psi_0 is None:
        psi_0 = (
            1 / 10 * np.identity(num_groups * (num_candidates - 1))
        )  # relates to prior precision
    if k_0_inv is None:
        k_0_inv = 1 / (0.5) * np.identity(num_groups * (num_candidates - 1))  # same size as Sigma
    if mu_0 is None:
        votes_all_precincts = vote_counts.sum(axis=0)
        log_ratio_support = np.log(
            votes_all_precincts[0:-1] / votes_all_precincts[-1]
        )  # assume each group supports according to the total vote split
        mu_0 = np.tile(log_ratio_support, num_groups)  # shape r * (c-1)

    samples = greiner_quinn_gibbs_sample(
        group_counts,
        vote_counts,
        num_samples,
        nu_0,
        psi_0,
        k_0_inv,
        mu_0,
        gamma=gamma,
        burnin=burnin,
    )
    # convert to InferenceData object
    for var_name in samples.keys():  # pylint: disable=consider-iterating-dictionary
        samples[var_name] = np.expand_dims(samples[var_name], 0)  # add an axis for chain
    samples["b"] = samples.pop("theta")  # changing variable name for vote prefernces to match pyei
    idata = az.from_dict(samples)

    return idata


def greiner_quinn_gibbs_sample(  # pylint: disable=too-many-locals
    group_counts, vote_counts, num_samples, nu_0, psi_0, k_0_inv, mu_0, gamma=0.1, burnin=0
):
    """
    group_counts: ndarray
        num_precincts x r gives number of people for each of r groups
        in num_precincts precincts
    vote_counts: ndarray
        num_precincts x c gives number of votes for each of c candidates
        in num_precincts precincts
    num_samples: int
        number of MCMC samples to draw using the Gibbs sampler
    nu_0: int
        hyperparameter (df) scalar
        roughly interpretable as number of pseudodistricts for prior
    psi_0: ndarray
        hyperparameter (scale) - square matrix r * (c - 1) x r * (c - 1)
    k_0_inv:
        hyperparameter - square matrix r * (c - 1) x r * (c - 1)
        Precision for prior distribution over mu, the mean of the distribution of omega
        mu | mu_0, k_0 ~ N(mu_0, k_0) = N(mu_0, k_0_inv^{-1})
        omega | mu, Sigma ~ N(mu, Sigma)
    mu_0: ndarray
        vector of length r * (c - 1)
        Governs prior distribution over mu, the mean of the distribution of omega
        mu | mu_0, k_0 ~ N(mu_0, k_0)
        omega | mu, Sigma ~ N(mu, Sigma)
    gamma: float
        parameter of proposal dist in the Metropolis step for sampling theta
    burnin: int
        the number of samples at the start of the MCMC chain to be discarded

    Note:
    -----
    Variable names follow those Greiner and Quinn, R x C ecological inference:
    bounds, correlations, flexibility and transparency of assumptions (2009)
    """
    # TODO: gamma set in initial runs?

    num_precincts = group_counts.shape[0]
    if np.all(num_precincts != vote_counts.shape[0]):
        raise ValueError(
            "group_counts and vote_counts must both have first dim of length num_precincts"
        )
    if burnin >= num_samples:
        raise ValueError(f"num_samples is only {num_samples} but burn-in is {burnin}")

    precinct_pops = vote_counts.sum(axis=1)
    _, num_candidates = vote_counts.shape
    _, num_groups = group_counts.shape

    # set initial values
    mu_samp = mu_0
    Sigma_samp = st.invwishart.rvs(df=nu_0, scale=psi_0)
    omega_samp = np.zeros((num_precincts, num_groups * (num_candidates - 1)))
    print(num_groups, num_candidates)
    theta_samp = omega_to_theta(omega_samp, num_groups, num_candidates)
    internal_cell_counts_samp = get_initial_internal_count_sample(
        group_counts, vote_counts, precinct_pops
    )

    # variables for storing all samples
    internal_cell_counts_samples = np.empty(
        (num_samples - burnin, num_precincts, num_groups, num_candidates)
    )
    theta_samples = np.empty((num_samples - burnin, num_precincts, num_groups, num_candidates))
    mu_samples = np.empty((num_samples - burnin, num_groups * (num_candidates - 1)))
    sigma_samples = np.empty(
        (num_samples - burnin, num_groups * (num_candidates - 1), num_groups * (num_candidates - 1))
    )

    for i in tqdm(range(num_samples)):
        # (a) sample internal cell counts
        internal_cell_counts_samp = sample_internal_cell_counts(
            theta_samp, internal_cell_counts_samp
        )

        # (b) sample theta using Metropolis-Hastings
        theta_samp = sample_theta(
            internal_cell_counts_samp, theta_samp, omega_samp, mu_samp, Sigma_samp, gamma
        )

        omega_samp = theta_to_omega(theta_samp)
        # TODO: IS THE NEXT LINE UNNECESSARY?
        omega_samp = omega_samp.reshape((num_precincts, num_groups * (num_candidates - 1)))

        # (c) sample mu and sigma given omega (or, equivalently, given theta)
        mu_samp = sample_mu(omega_samp, Sigma_samp, k_0_inv, mu_0, num_precincts)
        Sigma_samp = sample_Sigma(omega_samp, mu_samp, nu_0, psi_0)

        # save samples after burn-in
        if i >= burnin:
            internal_cell_counts_samples[i - burnin, :, :, :] = internal_cell_counts_samp
            theta_samples[i - burnin, :, :, :] = theta_samp
            mu_samples[i - burnin, :] = mu_samp
            sigma_samples[i - burnin, :, :] = Sigma_samp

    return {
        "theta": theta_samples,
        "counts": internal_cell_counts_samples,
        "mu": mu_samples,
        "Sigma": sigma_samples,
    }


def sample_Sigma(omega, mu, nu_0, psi_0):
    """
    Parameters:
    -----------
    omega: ndarray
        num_precints x (r * (c - 1))
        A transformation of the voting preferences given by theta,
        omega(i, r', c') = log(theta(i,r',c')/theta(i,r, num_candidates)),
    mu: array
        vector of length r * (c - 1)
        mean of the normal distribution that governs omega
        omega | mu, Sigma ~ N(mu, Sigma)
        where omega(i, r', c') = log(theta(i,r',c')/theta(i,r, c'')),
        theta are the voting preferences of each group and candidate in
        each precinct, and c'' is some reference column/candidate
        (here the last column)
    nu_0: float
        hyperparameter (deg of freedom) - scalar
    psi_0: n
        darray:hyperparameter (scale)
        square matrix r * (c - 1) x r * (c - 1)

    Returns:
    --------
    Sigma: ndarray
        square matrix r * (c - 1) x r * (c - 1), the covariance of omega
    """
    num_precincts = omega.shape[0]
    nu_n = nu_0 + num_precincts
    psi_n = psi_0 + ((omega - mu) @ (omega - mu).T).sum()  # sum over precincts

    Sigma = st.invwishart.rvs(nu_n, psi_n)
    return Sigma


def sample_mu(omega, Sigma, k_0_inv, mu_0, num_precincts):
    """
    omega: num_precints x (r * (c - 1))
    Sigma: ndarray
        square matrix r * (c - 1) x r * (c - 1), the covariance of omega
    k_0_inv is hyperparameter - square matrix r * (c - 1) x r * (c - 1)
    mu_0 : array
        vector of length r * (c - 1)
        mean of the prior distribution for mu, mu the mean of of the normal
        distribution that governs omega
        mu | mu_0, k_0 ~ N(mu_0, k_0)
        omega | mu, Sigma ~ N(mu, Sigma)
        where omega(i, r', c') = log(theta(i,r',c')/theta(i,r, c'')),
        theta are the voting preferences of each group and candidate in
        each precinct, and c'' is some reference column/candidate
        (here the last column)
    num_precincts: int
        The number of precincts

    Returns:
    mu: a vector of length r * (c-1)
    """

    Sigma_inv = np.linalg.inv(Sigma)
    mean_omega = np.mean(omega, axis=0)
    mu_n = (np.linalg.inv(k_0_inv + num_precincts * Sigma_inv)) @ (
        k_0_inv @ mu_0.T + num_precincts * Sigma_inv @ mean_omega
    )
    Sigma_n_inv = k_0_inv + num_precincts * Sigma_inv
    mu = st.multivariate_normal.rvs(mean=mu_n, cov=np.linalg.inv(Sigma_n_inv))

    return mu


def proposal_dist_generate_sample(mu_samp, Sigma_samp, gamma, num_precincts, deg_freedom=4):
    """
    Grainer and Quinn use a t_4(mu_t, gamma * Sigma) proposal dist with gamma
    set during inital runs - this generates an omega sample, which they transform
    back to theta space

    Returns:
    --------
    omega_proposed:n num_precincts * (r * (c-1)
    """
    omega_proposed = st.multivariate_t.rvs(
        mu_samp, gamma * Sigma_samp, df=deg_freedom, size=num_precincts
    )
    return omega_proposed


def log_unnormalized_pdf(theta, omega, internal_cell_counts_samp, mu_samp, Sigma_samp, tol=0.01):
    """pdf proportional t to the product of lines (4)-(6) and (10) and (11) in G&Q
    0 if thetas don't sum to 1 across rows

    Sigma_samp: ndarray
        square matrix r * (c - 1) x r * (c - 1), a sample of the covariance of omega
    tol: float
        how far can the sum of theta be from 1
    theta: ndarray
        num_precints x r x c
    """

    # Lines (10) - (11) are this check
    if not np.all(abs(theta.sum(axis=2) - 1) < tol):  # CHECK IF THETAS SUM TO 1 across rows
        return -np.inf  # if not, prob is zero, so log_prob is -inf

    else:
        line_4_and_6 = np.apply_over_axes(
            np.sum, (internal_cell_counts_samp - 1) * np.log(theta), [1, 2]
        ).flatten()  # sum over rows and columns
        line_5 = -0.5 * np.linalg.det(Sigma_samp) - 0.5 * (
            (omega - mu_samp) @ np.linalg.inv(Sigma_samp) * (omega - mu_samp)
        ).sum(axis=1)
        return (line_4_and_6 + line_5).sum()  # sum over precincts


def theta_to_omega(theta):
    """
    Parameters:
    -----------
    theta: ndarray
        num_precints x r x c

    Returns:
    --------
    omega: ndarray
        num_precincts x r x (c-1)
    """
    return np.log(theta[:, :, :-1]) - np.log(theta[:, :, [-1]])


def sample_theta(internal_cell_counts_samp, theta_prev, omega_prev, mu_samp, Sigma_samp, gamma):
    """
    Use a Metropolis-Hastings step to sample theta

    internal_cell_counts_samp: ndarray
    theta_prev: ndarray
    omega_prev: ndarray
    mu_samp:
    Sigma_samp: ndarray
        square matrix r * (c - 1) x r * (c - 1), a sample of the covariance of omega
    gamma: float
        parameter of proposal dist in the Metropolis step

    """
    # MH
    num_precincts, r, c = internal_cell_counts_samp.shape

    omega_proposed = proposal_dist_generate_sample(
        mu_samp, Sigma_samp, gamma, num_precincts, deg_freedom=4
    )

    theta_proposed = omega_to_theta(omega_proposed, r, c)

    log_unnormalized_pdf_proposed = log_unnormalized_pdf(
        theta_proposed, omega_proposed, internal_cell_counts_samp, mu_samp, Sigma_samp
    )
    log_unnormalized_pdf_prev = log_unnormalized_pdf(
        theta_prev, omega_prev, internal_cell_counts_samp, mu_samp, Sigma_samp
    )

    # calculate acceptance prbability
    log_acc_prob = log_unnormalized_pdf_proposed - log_unnormalized_pdf_prev
    log_acc_prob = np.min([log_acc_prob, 0])
    acc_prob = np.exp(log_acc_prob)

    unif_samp = st.uniform.rvs(size=1)
    if unif_samp < acc_prob:
        return theta_proposed
    else:
        return theta_prev


def omega_to_theta(omega, r, c):
    """
    theta: num_precints x r x c
    omega: num_precincts x (r * (c-1))

    Note:
    -----
    SIDE EFFECT: RESHAPES OMEGA
    """
    num_precincts = omega.shape[0]
    omega = omega.reshape(num_precincts, r, c - 1)
    theta_last = 1.0 / (np.exp(omega).sum(axis=2) + 1)  # num_precincts x r x 1
    theta_other = (np.exp(omega).T * theta_last.T).T  # num_precincts x r x c - 1
    theta_last_ext = np.expand_dims(theta_last, axis=2)
    return np.concatenate((theta_other, theta_last_ext), axis=2)


@njit(parallel=True)
def sample_internal_cell_counts(theta_samp, prev_internal_counts_samp):
    """
    group_counts: num_precincts x r
    vote_counts: num_precincts x c
    theta: num_precints x r x c
    prev_internal_counts_samp: num_precincts x r x c

    @TODO VECTORIZE
    """
    num_precincts, num_groups, num_candidates = prev_internal_counts_samp.shape

    for i in prange(num_precincts):  # pylint: disable=not-an-iterable
        for r in range(num_groups - 1):
            for r_prime in range(r + 1, num_groups):
                for c in range(num_candidates - 1):
                    for c_prime in range(c + 1, num_candidates):
                        n1 = (  # pylint: disable=invalid-name
                            prev_internal_counts_samp[i, r, c]
                            + prev_internal_counts_samp[i, r, c_prime]
                        )  # n1 gives row total in row r
                        n2 = (  # pylint: disable=invalid-name
                            prev_internal_counts_samp[i, r_prime, c]
                            + prev_internal_counts_samp[i, r_prime, c_prime]
                        )  # n2 row total in row r_prime
                        m1 = (  # pylint: disable=invalid-name
                            prev_internal_counts_samp[i, r, c]
                            + prev_internal_counts_samp[i, r_prime, c]
                        )  # m1 gives column total in column c
                        pi1 = theta_samp[i, r, c]
                        pi2 = theta_samp[i, r_prime, c]
                        psi = (pi1 * (1 - pi2)) / (pi2 * (1 - pi1))
                        r_c_count = non_central_hypergeometric_sample(
                            n1, n2, m1, psi
                        )  # sample for the r, c internal count

                        # update prev_internal counts in the 2 x 2 subarray
                        prev_internal_counts_samp[i, r, c] = r_c_count
                        prev_internal_counts_samp[i, r, c_prime] = n1 - r_c_count
                        prev_internal_counts_samp[i, r_prime, c] = m1 - r_c_count
                        prev_internal_counts_samp[i, r_prime, c_prime] = (
                            n2 - prev_internal_counts_samp[i, r_prime, c]
                        )
    return prev_internal_counts_samp


@njit(parallel=True, nopython=False)
def get_initial_internal_count_sample(group_counts, vote_counts, precinct_pops):
    """
    Sets an initial value of internal counts that is compatible with the
    observed vote and group counts

    Parameters:
    -----------
    group_counts: ndarray
        num_precincts x r. Give count of people in each group within each
        precinct.
    vote_counts: ndarray
        num_precincts x c. Gives count of votes for each candidate within each
        precinct
    precinct_pops: vector of length num_precincts

    Returns:
    --------
        internal_counts_samp: ndarray
            num_precincts x r x c
            Gives a set of "internal counts" in the table of results
            for each precinct - gives a set of counts for people
            within each group that voted for each candidates that
            is possible given the marginal sums
            vote_counts and group_counts

    Notes:
    ------
    Within each of rows up to row num_groups-1, for c=0,..num_candidates-1
    and samples according to a binomial with
    p=vote_counts[i, c] / precinct_pops[i],
    making sure not to exceed the remaining counts left to assign in the
    column and row.
    Keep track of remaining counts in each row and column and use that
    to fill the last row and column.
    """
    num_precincts, num_groups = group_counts.shape
    _, num_candidates = vote_counts.shape
    internal_counts_samp = np.empty((num_precincts, num_groups, num_candidates), dtype=np.int16)

    group_counts_remaining = group_counts.copy()
    vote_counts_remaining = vote_counts.copy()

    for i in prange(num_precincts):  # pylint: disable=not-an-iterable
        for r in range(num_groups - 1):
            for c in range(num_candidates - 1):
                count_for_binom = np.round(
                    group_counts[i, r]
                    * (vote_counts[i, c] + vote_counts[i, c + 1])
                    / precinct_pops[i]
                )
                prop_for_binom = vote_counts[i, c] / precinct_pops[i]
                samp = np.random.binomial(n=int(count_for_binom), p=int(prop_for_binom))

                samp = min(samp, vote_counts_remaining[i, c])
                samp = min(samp, group_counts_remaining[i, r])
                group_counts_remaining[i, r] = group_counts_remaining[i, r] - samp
                vote_counts_remaining[i, c] = vote_counts_remaining[i, c] - samp
                internal_counts_samp[i, r, c] = samp
            internal_counts_samp[i, r, num_candidates - 1] = group_counts_remaining[i, r]
            vote_counts_remaining[i, num_candidates - 1] = (
                vote_counts_remaining[i, num_candidates - 1]
                - internal_counts_samp[i, r, num_candidates - 1]
            )
        internal_counts_samp[i, num_groups - 1, :] = vote_counts_remaining[i, :]

    return internal_counts_samp
