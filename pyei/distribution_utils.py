"""Port of R code for Noncentral Hypergeometric distribution.

Adapted from R code published in conjunction with:
Liao, J.G. And Rosen, O. (2001) Fast and Stable Algorithms for Computing and
Sampling from the Noncentral Hypergeometric Distribution.  The American
Statistician 55, 366-369.

Used in Greiner-Quinn method Gibbs sampler.
"""

import math

import numpy as np
from numba import jit


@jit
def _r_function(n1: int, n2: int, m1: int, psi: float, i: int) -> float:
    """The function r defined in Liao and Rosen 2001"""
    return (n1 - i + 1) * (m1 - i + 1) / (i * (n2 - m1 + i)) * psi


@jit
def _sample_low_to_high(
    lower: int, ran: float, pi: np.ndarray, shift: int, uu: int
) -> int:
    for i in range(lower, uu + 1):
        if ran <= pi[i + shift]:
            return i
        ran = ran - pi[i + shift]
    return uu  # fallback


@jit
def _sample_high_to_low(
    upper: int, ran: float, pi: np.ndarray, shift: int, ll: int
) -> int:
    for i in range(upper, ll - 1, -1):
        if ran <= pi[i + shift]:
            return i
        ran = ran - pi[i + shift]
    return ll  # fallback


@jit
def non_central_hypergeometric_sample(n1: int, n2: int, m1: int, psi: float) -> int:
    """Sample from noncentral hypergeometric distribution.

    Following the methods of Liao and Rosen, 2001.

    Parameters
    ----------
    n1 : int
        Number of items in first group
    n2 : int
        Number of items in second group
    m1 : int
        Number of items to sample
    psi : float
        Odds ratio parameter

    Notes:
    -----
    If
    y1 ~ Binom(n1, pi1)
    y2 ~ Binom(n2, pi2)
    psi = pi1 (1-pi2) / pi2 (1-pi1)

    this the nchg distribution governs with parameters n1, n2, pis, m1
    is the distribution for y1 | y1 + y2 = m1
    """
    ll = max(0, m1 - n2)
    uu = min(n1, m1)

    a = psi - 1
    b = -((m1 + n1 + 2) * psi + n2 - m1)
    c = psi * (n1 + 1) * (m1 + 1)
    q = -(b + np.sign(b) * np.sqrt(b * b - 4 * a * c)) / 2
    mode_candidate = math.trunc(c / q)
    if (uu >= mode_candidate) and (ll <= mode_candidate):
        mode = mode_candidate
    else:
        mode = math.trunc(q / a)

    # calculate density
    pi = np.ones(uu - ll + 1, dtype=np.float32)

    if mode < uu:
        r = _r_function(n1, n2, m1, psi, np.arange(mode + 1, uu + 1))
        pi[(mode + 1 - ll) : (uu - ll + 1)] = np.cumprod(r)

    if mode > ll:
        r = 1 / _r_function(
            n1,
            n2,
            m1,
            psi,
            np.flip(np.arange(ll + 1, mode + 1)),
        )
        pi[0 : (mode - ll)] = np.flip(np.cumprod(r))
    density = pi / pi.sum()

    ran = np.random.uniform(0, 1)
    pi = density

    if mode == ll:
        return _sample_low_to_high(ll, ran, pi, -ll, uu)
    if mode == uu:
        return _sample_high_to_low(uu, ran, pi, -ll, ll)
    if ran < pi[mode - ll]:
        return mode
    ran = ran - pi[mode - ll]
    lower = mode - 1
    upper = mode + 1

    while True:
        if pi[upper - ll] >= pi[lower - ll]:
            if ran < pi[upper - ll]:
                return upper
            ran = ran - pi[upper - ll]
            if upper == uu:
                samp = _sample_high_to_low(lower, ran, pi, -ll, ll)
                return samp
            upper = upper + 1

        else:
            if ran < pi[lower - ll]:
                return lower
            ran = ran - pi[lower - ll]
            if lower == ll:
                samp = _sample_low_to_high(upper, ran, pi, -ll, uu)
                return samp
            lower = lower - 1
