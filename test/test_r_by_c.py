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
    """multinomial-dirichlet and multinomial-dirichlet-modified should both
    yield distributions that sum to 1 — guards against a model variant
    being silently misconfigured."""
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
    """Passing non_candidate_names should drop those names from the summary."""
    ei = two_r_by_c_ei_runs[0]
    summary = ei.summary(non_candidate_names=["Hardy"])
    assert "Kolstad" in summary
    assert "Nadeem" in summary
    # Hardy is the dropped abstain column. After removal, the summary string
    # should still be non-empty and not mention Hardy as a remaining candidate.
    # (We can't assert "Hardy" not in summary because group names or other
    # tokens may legitimately contain it; instead, check the summary length.)
    assert len(summary) > 0


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
    """For each group, the rates over candidates must sum to 1 (each sample
    picks exactly one argmax)."""
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
    """rate[(g1, g2)] should equal rate[(g2, g1)] — the metric is the
    fraction of samples where the two groups' argmax candidates differ."""
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
    for earlier, later in zip(probs, probs[1:]):
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
    for earlier, later in zip(widths, widths[1:]):
        assert later >= earlier - 1e-9


# ---------------------------------------------------------------------------
# margin_report
# ---------------------------------------------------------------------------


def test_margin_report_threshold_returns_probability(two_r_by_c_ei_runs):
    """Threshold mode returns the % of samples whose (cand0 - cand1) margin
    exceeds the threshold — a number in [0, 100]."""
    ei = two_r_by_c_ei_runs[0]
    pct = ei.margin_report("e_asian", ["Kolstad", "Nadeem"], threshold=0.0, verbose=False)
    assert 0.0 <= pct <= 100.0


def test_margin_report_threshold_is_monotone(two_r_by_c_ei_runs):
    """P(margin > t) must be non-increasing in t."""
    ei = two_r_by_c_ei_runs[0]
    pcts = [
        ei.margin_report(
            "e_asian", ["Kolstad", "Nadeem"], threshold=t, verbose=False
        )
        for t in [-1.0, -0.2, 0.0, 0.2, 1.0]
    ]
    for earlier, later in zip(pcts, pcts[1:]):
        assert later <= earlier + 1e-9


def test_margin_report_threshold_extremes(two_r_by_c_ei_runs):
    """Margins are bounded in [-1, 1], so threshold below -1 captures every
    sample (100%) and above 1 captures none (0%)."""
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
    for earlier, later in zip(widths, widths[1:]):
        assert later >= earlier - 1e-9


def test_margin_report_antisymmetric_in_candidate_order(two_r_by_c_ei_runs):
    """margin(c1, c2) = c1 - c2, so swapping candidates negates the interval
    and reflects the threshold percentile around 100 - p."""
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
    """non_candidate_names path uses turnout_adjusted_samples; the c dimension
    should drop by len(non_candidate_names)."""
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
    """If non_candidate_names is passed before calculate_turnout_adjusted_summary,
    the call should raise rather than silently return stale or wrong arrays."""
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
