"""Direct behavioural tests for RowByColumnEI.

Plotting-driven tests live in test_r_by_c_plotting.py. This file targets
the inference and reporting surface independently of any plot rendering.
"""

import numpy as np
import pytest

from pyei.r_by_c import RowByColumnEI

# All tests in this module exercise a fitted RowByColumnEI (PyMC/numpyro
# NUTS sampling). The unsupported-model-name test is the only exception
# and is marked individually as fast below.
pytestmark = pytest.mark.slow

# ---------------------------------------------------------------------------
# Shape & marginal invariants
# ---------------------------------------------------------------------------


def test_sim_trace_posterior_b_shape(two_r_by_c_ei_runs):
    """posterior['b'] is chain x draw x num_precincts x r x c."""
    ei = two_r_by_c_ei_runs[0]
    b = ei.sim_trace["posterior"]["b"].values
    assert b.ndim == 5
    num_precincts = len(ei.precinct_pops)
    r, c = ei.num_groups_and_num_candidates
    assert b.shape[2] == num_precincts
    assert b.shape[3] == r
    assert b.shape[4] == c


def test_sampled_voting_prefs_shape(two_r_by_c_ei_runs):
    """sampled_voting_prefs is num_samples x r x c."""
    ei = two_r_by_c_ei_runs[0]
    r, c = ei.num_groups_and_num_candidates
    assert ei.sampled_voting_prefs.ndim == 3
    assert ei.sampled_voting_prefs.shape[1] == r
    assert ei.sampled_voting_prefs.shape[2] == c


def test_sampled_voting_prefs_sum_to_one_per_group(two_r_by_c_ei_runs):
    """Across candidates, each group's vote shares must sum to 1 in every sample."""
    ei = two_r_by_c_ei_runs[0]
    sums = ei.sampled_voting_prefs.sum(axis=2)  # num_samples x r
    np.testing.assert_allclose(sums, 1.0, atol=1e-6)


def test_sampled_voting_prefs_in_unit_interval(two_r_by_c_ei_runs):
    """Every entry is a probability."""
    prefs = two_r_by_c_ei_runs[0].sampled_voting_prefs
    assert prefs.min() >= 0.0
    assert prefs.max() <= 1.0


def test_posterior_mean_matches_sampled_mean(two_r_by_c_ei_runs):
    ei = two_r_by_c_ei_runs[0]
    np.testing.assert_allclose(
        ei.posterior_mean_voting_prefs,
        ei.sampled_voting_prefs.mean(axis=0),
    )


def test_credible_interval_bounds_credible_mean(two_r_by_c_ei_runs):
    """For each (group, candidate), the 95% CI must bracket the posterior mean."""
    ei = two_r_by_c_ei_runs[0]
    means = ei.posterior_mean_voting_prefs
    ci = ei.credible_interval_95_mean_voting_prefs
    assert np.all(ci[..., 0] <= means + 1e-9)
    assert np.all(means - 1e-9 <= ci[..., 1])
    assert np.all(ci[..., 0] >= 0.0)
    assert np.all(ci[..., 1] <= 1.0)


# ---------------------------------------------------------------------------
# Model parametrisation
# ---------------------------------------------------------------------------


def test_both_model_variants_produce_valid_prefs(two_r_by_c_ei_runs):
    for ei in two_r_by_c_ei_runs[:2]:
        sums = ei.sampled_voting_prefs.sum(axis=2)
        np.testing.assert_allclose(sums, 1.0, atol=1e-6)


def test_unsupported_model_name_rejected(example_r_by_c_data):
    ei = RowByColumnEI(model_name="not-a-real-model")
    with pytest.raises(ValueError, match="not a supported model_name"):
        ei.fit(
            example_r_by_c_data["group_fractions"],
            example_r_by_c_data["votes_fractions"],
            example_r_by_c_data["precinct_pops"],
            example_r_by_c_data["demographic_group_names"],
            example_r_by_c_data["candidate_names"],
            draws=10,
            tune=10,
        )


# ---------------------------------------------------------------------------
# summary() content
# ---------------------------------------------------------------------------


def test_summary_mentions_all_group_and_candidate_names(two_r_by_c_ei_runs):
    ei = two_r_by_c_ei_runs[0]
    summary = ei.summary()
    for group in ei.demographic_group_names:
        assert group in summary, f"summary missing group {group!r}"
    for cand in ei.candidate_names:
        assert cand in summary, f"summary missing candidate {cand!r}"


def test_summary_reports_posterior_means_to_three_decimals(two_r_by_c_ei_runs):
    """The summary template formats means as '{:.3f}' — check at least one."""
    ei = two_r_by_c_ei_runs[0]
    summary = ei.summary()
    mean = ei.posterior_mean_voting_prefs[0][0]
    assert f"{mean:.3f}" in summary


def test_summary_with_non_candidate_names_uses_turnout_adjusted(two_r_by_c_ei_runs):
    """Passing non_candidate_names should drop those names from the summary.

    The fixture's demographic group names are ``ind``, ``e_asian``,
    ``non_asian`` — none contain the substring "Hardy" — so the dropped
    candidate must not appear anywhere in the rendered summary.
    """
    ei = two_r_by_c_ei_runs[0]
    summary = ei.summary(non_candidate_names=["Hardy"])
    assert "Kolstad" in summary
    assert "Nadeem" in summary
    assert "Hardy" not in summary


def test_summary_with_unknown_non_candidate_name_raises(two_r_by_c_ei_runs):
    """The ``summary()`` wrapper must surface the unknown-name ValueError
    from ``_calculate_turnout_adjusted_samples`` rather than silently
    rendering a partial summary or crashing on a downstream attribute.
    """
    ei = two_r_by_c_ei_runs[0]
    with pytest.raises(ValueError, match="non_candidate_names must be in candidate_names"):
        ei.summary(non_candidate_names=["NotAName"])


# ---------------------------------------------------------------------------
# Turnout-adjusted summary edge cases
# ---------------------------------------------------------------------------


def test_turnout_adjusted_summary_shape(two_r_by_c_ei_runs):
    ei = two_r_by_c_ei_runs[0]
    ei.calculate_turnout_adjusted_summary(["Hardy"])
    r, c = ei.num_groups_and_num_candidates
    # Adjusted samples drop the abstain column → (c - 1) candidates remain.
    assert ei.turnout_adjusted_samples.shape[-1] == c - 1
    assert ei.turnout_adjusted_posterior_mean_voting_prefs.shape == (r, c - 1)


def test_turnout_adjusted_samples_sum_to_one(two_r_by_c_ei_runs):
    """After removing the abstain column and renormalising, remaining shares sum to 1."""
    ei = two_r_by_c_ei_runs[0]
    ei.calculate_turnout_adjusted_summary(["Hardy"])
    # turnout_adjusted_samples is num_samples x num_precincts x r x (c-1)
    sums = ei.turnout_adjusted_samples.sum(axis=-1)
    np.testing.assert_allclose(sums, 1.0, atol=1e-6)


def test_turnout_adjusted_summary_unknown_candidate_raises(two_r_by_c_ei_runs):
    """Passing a name not in candidate_names should raise (.index() raises ValueError)."""
    ei = two_r_by_c_ei_runs[0]
    with pytest.raises(ValueError):
        ei.calculate_turnout_adjusted_summary(["NotAName"])


# ---------------------------------------------------------------------------
# candidate_of_choice / polarization reports
# ---------------------------------------------------------------------------


def test_candidate_of_choice_report_rates_sum_to_one_per_group(
    two_r_by_c_ei_runs, capsys
):
    """For each group, the rates over candidates must sum to 1."""
    ei = two_r_by_c_ei_runs[0]
    rates = ei.candidate_of_choice_report(verbose=False)
    capsys.readouterr()  # discard any stray prints

    for group in ei.demographic_group_names:
        total = sum(rates[(group, c)] for c in ei.candidate_names)
        np.testing.assert_allclose(total, 1.0, atol=1e-9)


def test_candidate_of_choice_report_drops_non_candidates(two_r_by_c_ei_runs):
    """non_candidate_names should remove that key from the dictionary."""
    ei = two_r_by_c_ei_runs[0]
    rates = ei.candidate_of_choice_report(verbose=False, non_candidate_names=["Hardy"])
    for group in ei.demographic_group_names:
        assert (group, "Hardy") not in rates
        assert (group, "Kolstad") in rates
        assert (group, "Nadeem") in rates


def test_candidate_of_choice_polarization_is_symmetric_in_groups(two_r_by_c_ei_runs):
    ei = two_r_by_c_ei_runs[0]
    rates = ei.candidate_of_choice_polarization_report(verbose=False)
    groups = ei.demographic_group_names
    for g1 in groups:
        for g2 in groups:
            if g1 == g2:
                continue
            if (g1, g2) in rates and (g2, g1) in rates:
                np.testing.assert_allclose(rates[(g1, g2)], rates[(g2, g1)])


def test_polarization_report_threshold_is_monotone(two_r_by_c_ei_runs):
    """P(|g1_pref - g2_pref| > t) is monotone non-increasing in t."""
    ei = two_r_by_c_ei_runs[0]
    groups = ["e_asian", "non_asian"]
    candidate = "Kolstad"
    probs = [
        ei.polarization_report(groups, candidate, threshold=t)
        for t in [0.0, 0.1, 0.2, 0.4, 0.8]
    ]
    for earlier, later in zip(probs[:-1], probs[1:], strict=True):
        assert later <= earlier + 1e-9


def test_polarization_report_percentile_widens_with_coverage(two_r_by_c_ei_runs):
    """Wider percentile coverage ⇒ at-least-as-wide interval."""
    ei = two_r_by_c_ei_runs[0]
    groups = ["e_asian", "non_asian"]
    candidate = "Kolstad"
    widths = []
    for p in [50, 80, 90, 95, 99]:
        lo, hi = ei.polarization_report(groups, candidate, percentile=p)
        widths.append(hi - lo)
    for earlier, later in zip(widths[:-1], widths[1:], strict=True):
        assert later >= earlier - 1e-9


# ---------------------------------------------------------------------------
# margin_report
# ---------------------------------------------------------------------------


def test_margin_report_threshold_returns_probability(two_r_by_c_ei_runs):
    ei = two_r_by_c_ei_runs[0]
    pct = ei.margin_report(
        "e_asian", ["Kolstad", "Nadeem"], threshold=0.0, verbose=False
    )
    assert 0.0 <= pct <= 100.0


def test_margin_report_threshold_is_monotone(two_r_by_c_ei_runs):
    """P(margin > t) must be non-increasing in t."""
    ei = two_r_by_c_ei_runs[0]
    pcts = [
        ei.margin_report("e_asian", ["Kolstad", "Nadeem"], threshold=t, verbose=False)
        for t in [-1.0, -0.2, 0.0, 0.2, 1.0]
    ]
    for earlier, later in zip(pcts[:-1], pcts[1:], strict=True):
        assert later <= earlier + 1e-9


def test_margin_report_threshold_extremes(two_r_by_c_ei_runs):
    ei = two_r_by_c_ei_runs[0]
    assert ei.margin_report(
        "e_asian", ["Kolstad", "Nadeem"], threshold=-2.0, verbose=False
    ) == pytest.approx(100.0)
    assert ei.margin_report(
        "e_asian", ["Kolstad", "Nadeem"], threshold=2.0, verbose=False
    ) == pytest.approx(0.0)


def test_margin_report_percentile_returns_ordered_interval(two_r_by_c_ei_runs):
    ei = two_r_by_c_ei_runs[0]
    lo, hi = ei.margin_report(
        "e_asian", ["Kolstad", "Nadeem"], percentile=95, verbose=False
    )
    assert lo <= hi
    assert -1.0 <= lo <= 1.0
    assert -1.0 <= hi <= 1.0


def test_margin_report_percentile_widens_with_coverage(two_r_by_c_ei_runs):
    """Higher percentile coverage ⇒ at-least-as-wide interval."""
    ei = two_r_by_c_ei_runs[0]
    widths = []
    for p in [50, 80, 90, 95, 99]:
        lo, hi = ei.margin_report(
            "e_asian", ["Kolstad", "Nadeem"], percentile=p, verbose=False
        )
        widths.append(hi - lo)
    for earlier, later in zip(widths[:-1], widths[1:], strict=True):
        assert later >= earlier - 1e-9


def test_margin_report_antisymmetric_in_candidate_order(two_r_by_c_ei_runs):
    ei = two_r_by_c_ei_runs[0]
    lo_ab, hi_ab = ei.margin_report(
        "e_asian", ["Kolstad", "Nadeem"], percentile=90, verbose=False
    )
    lo_ba, hi_ba = ei.margin_report(
        "e_asian", ["Nadeem", "Kolstad"], percentile=90, verbose=False
    )
    np.testing.assert_allclose(lo_ba, -hi_ab, atol=1e-9)
    np.testing.assert_allclose(hi_ba, -lo_ab, atol=1e-9)


def test_margin_report_unknown_candidate_raises(two_r_by_c_ei_runs):
    ei = two_r_by_c_ei_runs[0]
    with pytest.raises(ValueError, match="candidate names"):
        ei.margin_report(
            "e_asian", ["Kolstad", "NotAName"], threshold=0.0, verbose=False
        )


def test_margin_report_unknown_group_raises(two_r_by_c_ei_runs):
    ei = two_r_by_c_ei_runs[0]
    with pytest.raises(ValueError, match="group name"):
        ei.margin_report(
            "not_a_group", ["Kolstad", "Nadeem"], threshold=0.0, verbose=False
        )


def test_margin_report_requires_threshold_or_percentile(two_r_by_c_ei_runs):
    """Passing neither argument should raise (the docstring requires exactly one)."""
    ei = two_r_by_c_ei_runs[0]
    with pytest.raises(ValueError):
        ei.margin_report("e_asian", ["Kolstad", "Nadeem"], verbose=False)


# ---------------------------------------------------------------------------
# precinct_level_estimates
# ---------------------------------------------------------------------------


def test_precinct_level_estimates_shapes(two_r_by_c_ei_runs):
    """Returns (num_precincts x r x c, num_precincts x r x c x 2)."""
    ei = two_r_by_c_ei_runs[0]
    means, intervals = ei.precinct_level_estimates()
    num_precincts = len(ei.precinct_pops)
    r, c = ei.num_groups_and_num_candidates
    assert means.shape == (num_precincts, r, c)
    assert intervals.shape == (num_precincts, r, c, 2)


def test_precinct_level_estimates_sum_to_one_per_group(two_r_by_c_ei_runs):
    """For each (precinct, group), the candidate means sum to 1."""
    means, _ = two_r_by_c_ei_runs[0].precinct_level_estimates()
    np.testing.assert_allclose(means.sum(axis=2), 1.0, atol=1e-6)


def test_precinct_level_estimates_intervals_bracket_means(two_r_by_c_ei_runs):
    means, intervals = two_r_by_c_ei_runs[0].precinct_level_estimates()
    lower = intervals[..., 0]
    upper = intervals[..., 1]
    assert np.all(lower <= means + 1e-9)
    assert np.all(means - 1e-9 <= upper)
    assert np.all(lower >= 0.0)
    assert np.all(upper <= 1.0)


def test_precinct_level_estimates_turnout_adjusted_drops_abstain_column(
    two_r_by_c_ei_runs,
):
    ei = two_r_by_c_ei_runs[0]
    ei.calculate_turnout_adjusted_summary(["Hardy"])
    means, intervals = ei.precinct_level_estimates(non_candidate_names=["Hardy"])
    r, c = ei.num_groups_and_num_candidates
    num_precincts = len(ei.precinct_pops)
    assert means.shape == (num_precincts, r, c - 1)
    assert intervals.shape == (num_precincts, r, c - 1, 2)
    # After renormalisation, each (precinct, group) again sums to 1 over remaining cands.
    np.testing.assert_allclose(means.sum(axis=2), 1.0, atol=1e-6)


def test_precinct_level_estimates_turnout_adjusted_requires_precompute(
    example_r_by_c_data,
):
    ei = RowByColumnEI(model_name="multinomial-dirichlet")
    ei.fit(
        example_r_by_c_data["group_fractions"],
        example_r_by_c_data["votes_fractions"],
        example_r_by_c_data["precinct_pops"],
        example_r_by_c_data["demographic_group_names"],
        example_r_by_c_data["candidate_names"],
        draws=20,
        tune=20,
        random_seed=0,
    )
    with pytest.raises(RuntimeError, match="Turnout adjusted samples"):
        ei.precinct_level_estimates(non_candidate_names=["Hardy"])


# ---------------------------------------------------------------------------
# Edge-case shapes: variations on (num_precincts, r, c, precinct_pops)
# ---------------------------------------------------------------------------


def _synthetic_r_by_c(num_precincts, r, c, precinct_pop, seed=0):
    """Build a self-consistent (group_frac, votes_frac, pops) trio of the requested shape."""
    rng = np.random.default_rng(seed)
    precinct_pops = np.full(num_precincts, precinct_pop, dtype=np.int64)

    def _integer_partition(num_categories):
        """Draw a (n_categories, n_precincts) matrix of nonneg ints with the correct pops."""
        probs = rng.dirichlet(np.ones(num_categories), size=num_precincts)
        counts = np.round(probs * precinct_pop).astype(np.int64)

        counts[:, -1] += precinct_pop - counts.sum(axis=1)
        if (counts < 0).any():
            # Pathological tiny-pop case: clamp and rebalance.
            counts = np.clip(counts, 0, None)
            counts[:, -1] += precinct_pop - counts.sum(axis=1)

        # need to transpose to get shape (num_categories, num_precincts) for the rest of the code
        return counts.T

    group_counts = _integer_partition(r)
    vote_counts = _integer_partition(c)
    group_fractions = group_counts / precinct_pop
    votes_fractions = vote_counts / precinct_pop
    return group_fractions, votes_fractions, precinct_pops


@pytest.mark.parametrize(
    "num_precincts, r, c, precinct_pop, seed",
    [
        # Single-precinct degenerate case — guards axis-0 reduction / slicing
        # bugs that wouldn't surface on multi-precinct fixtures.
        pytest.param(1, 2, 3, 500, 1, id="single_precinct"),
        # Sparse counts — small populations push observed vote shares toward
        # the binomial noise floor, stressing the model's identifiability.
        pytest.param(8, 2, 3, 15, 2, id="sparse_counts"),
        # Large candidate set — exercises dirichlet machinery beyond the
        # 3-candidate Santa Clara fixture.
        pytest.param(8, 2, 6, 500, 3, id="large_c"),
    ],
)
def test_fit_edge_case_shapes_produce_valid_voting_prefs(
    num_precincts, r, c, precinct_pop, seed
):
    """Each shape variant should fit cleanly and yield well-formed ``sampled_voting_prefs``.

    The ``sampled_voting_prefs is well-formed when it has the correct shape, is in [0, 1],
    and the entriessumming to 1 per group.


    Each case carries an explicit integer seed (not derived from the
    parametrize id) so the synthetic data is byte-identical across
    Python interpreter starts — Python's string ``hash()`` is randomised
    by ``PYTHONHASHSEED`` and would otherwise vary per process.
    """
    group_fractions, votes_fractions, precinct_pops = _synthetic_r_by_c(
        num_precincts, r, c, precinct_pop, seed=seed
    )

    ei = RowByColumnEI(model_name="multinomial-dirichlet")
    ei.fit(
        group_fractions,
        votes_fractions,
        precinct_pops,
        draws=80,
        tune=80,
        random_seed=0,
    )

    assert ei.sampled_voting_prefs is not None
    assert ei.sampled_voting_prefs.shape[1:] == (r, c)
    assert ei.sampled_voting_prefs.min() >= 0.0
    assert ei.sampled_voting_prefs.max() <= 1.0
    np.testing.assert_allclose(ei.sampled_voting_prefs.sum(axis=2), 1.0, atol=1e-6)
    # summary() must render without error for the new shape.
    assert isinstance(ei.summary(), str)


# ---------------------------------------------------------------------------
# Asymmetric (r != c) case
# ---------------------------------------------------------------------------


def test_asymmetric_run_has_expected_shapes(two_r_by_c_ei_runs):
    """The third run uses r=2, c=3 data — verify dims propagate correctly."""
    ei = two_r_by_c_ei_runs[2]
    r, c = ei.num_groups_and_num_candidates
    assert (r, c) == (2, 3)
    assert ei.sampled_voting_prefs.shape[1:] == (2, 3)
    sums = ei.sampled_voting_prefs.sum(axis=2)
    np.testing.assert_allclose(sums, 1.0, atol=1e-6)
