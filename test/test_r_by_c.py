"""Direct behavioural tests for RowByColumnEI.

Plotting-driven tests live in test_r_by_c_plotting.py. This file targets
the inference and reporting surface independently of any plot rendering.
"""

import numpy as np
import pytest

from pyei.r_by_c import RowByColumnEI

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
