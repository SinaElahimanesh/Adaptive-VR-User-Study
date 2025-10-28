from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

# Ensure local package import when running directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from cix_analysis.io import load_questionnaire, save_table
from cix_analysis.preprocess import build_tidy_frames
from cix_analysis.quantitative import (
    compute_descriptive_stats,
    compute_preference_distribution,
    run_friedman_by_task,
    compute_condition_effects,
    compute_correlations,
    compute_consistency_agreement,
    posthoc_condition_effects_pairwise,
    posthoc_friedman_within_condition,
    run_multinomial_choice_model,
    compute_cross_condition_adaptability,
)
from cix_analysis.qualitative import (
    extract_themes,
    compare_comments_by_condition,
    map_justifications_by_anchor_and_task,
    select_representative_quotes,
    compute_transformer_sentiment,
    aggregate_sentiment,
)
from cix_analysis.metrics import (
    compute_preference_strength_index,
    compute_anchor_adaptability,
    summarize_adaptability_within,
    summarize_adaptability_across,
)
from cix_analysis.mixed import (
    triangulate_findings,
)
from cix_analysis.visuals import (
    plot_preference_barcharts,
    plot_heatmaps,
    plot_radar_profiles,
    plot_sankey_planned_vs_rated,
    plot_wordclouds,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CIX Questionnaire analyses")
    parser.add_argument("--csv", required=True, help="Path to Questionnaire.csv")
    parser.add_argument("--outdir", default="outputs/questionnaire", help="Directory to write outputs")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    raw = load_questionnaire(args.csv)
    tidy = build_tidy_frames(raw)

    # Save tidy tables (only those used in report are saved downstream)

    # Quantitative analyses (unchanged)
    desc = compute_descriptive_stats(tidy.ratings_long)
    save_table(desc['by_condition'], os.path.join(args.outdir, "desc_by_condition.csv"))

    pref = compute_preference_distribution(tidy.rankings)
    save_table(pref, os.path.join(args.outdir, "preference_distribution.csv"))

    friedman = run_friedman_by_task(tidy.ratings_long)
    save_table(friedman, os.path.join(args.outdir, "friedman_tests.csv"))

    cond_effects = compute_condition_effects(tidy.ratings_long)
    save_table(cond_effects, os.path.join(args.outdir, "condition_effects.csv"))

    corrs = compute_correlations(tidy.participants, tidy.ratings_long, tidy.rankings)
    save_table(corrs['spearman'], os.path.join(args.outdir, "correlations_spearman.csv"))
    save_table(corrs['chi_square'], os.path.join(args.outdir, "correlations_chi_square.csv"))

    # Multinomial choice model (referenced in report)
    mnl = run_multinomial_choice_model(tidy.rankings, tidy.participants)
    if mnl is not None and not mnl.empty:
        save_table(mnl, os.path.join(args.outdir, "multinomial_choice_model.csv"))

    consistency = compute_consistency_agreement(tidy.ratings_long, tidy.rankings)
    save_table(consistency, os.path.join(args.outdir, "consistency_agreement.csv"))

    # Post-hoc tests (unchanged)
    posthoc_cond_rows = []
    for task in sorted(tidy.ratings_long['task'].dropna().unique()):
        for anchor in sorted(tidy.ratings_long['anchor'].dropna().unique()):
            ph = posthoc_condition_effects_pairwise(tidy.ratings_long, task, anchor)
            if not ph.empty:
                posthoc_cond_rows.append(ph)
    posthoc_cond = pd.concat(posthoc_cond_rows, ignore_index=True) if posthoc_cond_rows else pd.DataFrame()
    save_table(posthoc_cond, os.path.join(args.outdir, "posthoc_condition_pairwise.csv"))

    posthoc_friedman_rows = []
    for condition in sorted(tidy.ratings_long['condition'].dropna().unique()):
        for task in sorted(tidy.ratings_long['task'].dropna().unique()):
            phf = posthoc_friedman_within_condition(tidy.ratings_long, condition, task)
            if not phf.empty:
                posthoc_friedman_rows.append(phf)
    posthoc_fried = pd.concat(posthoc_friedman_rows, ignore_index=True) if posthoc_friedman_rows else pd.DataFrame()
    save_table(posthoc_fried, os.path.join(args.outdir, "posthoc_friedman_pairwise.csv"))

    # Qualitative analyses
    themes = extract_themes(tidy.comments)

    tfidf = compare_comments_by_condition(tidy.comments)
    save_table(tfidf, os.path.join(args.outdir, "condition_tfidf.csv"))

    justification = map_justifications_by_anchor_and_task(tidy.comments)

    quotes = select_representative_quotes(tidy.comments)
    save_table(quotes, os.path.join(args.outdir, "representative_quotes.csv"))

    # Transformer sentiment (state‑of‑the‑art) with fallback
    sent = compute_transformer_sentiment(tidy.comments)
    save_table(sent, os.path.join(args.outdir, "sentiment_transformer.csv"))
    sent_aggs = aggregate_sentiment(sent, tidy.comments)
    if 'by_condition_task' in sent_aggs:
        save_table(sent_aggs['by_condition_task'], os.path.join(args.outdir, "sentiment_by_condition_task.csv"))

    # Derived metrics
    psi = compute_preference_strength_index(tidy.ratings_long)
    save_table(psi, os.path.join(args.outdir, "preference_strength_index.csv"))

    adapt = compute_anchor_adaptability(tidy.ratings_long)
    save_table(adapt, os.path.join(args.outdir, "anchor_adaptability.csv"))
    adapt_within = summarize_adaptability_within(adapt)
    save_table(adapt_within, os.path.join(args.outdir, "anchor_adaptability_within_summary.csv"))

    cross_adapt = compute_cross_condition_adaptability(tidy.rankings, tidy.ratings_long)
    planned_summary = summarize_adaptability_across(cross_adapt['planned'], 'changes_planned_across_conditions')
    rated_summary = summarize_adaptability_across(cross_adapt['rated'], 'changes_rated_across_conditions')
    save_table(planned_summary, os.path.join(args.outdir, "adaptability_planned_across_summary.csv"))
    save_table(rated_summary, os.path.join(args.outdir, "adaptability_rated_across_summary.csv"))

    # Mixed methods
    tri = triangulate_findings(desc['by_condition'], justification, themes, tidy.comments)
    save_table(tri['theme_prevalence'], os.path.join(args.outdir, "triangulation_theme_prevalence.csv"))

    # Participant profiles output removed; computation skipped

    # Visualizations
    plot_preference_barcharts(pref, os.path.join(args.outdir, "figs"))
    plot_heatmaps(desc['by_condition'], os.path.join(args.outdir, "figs"))

    # Radar per condition for first participant
    if not tidy.participants.empty:
        pid = tidy.participants['participant_id'].iloc[0]
        for cond in sorted(tidy.ratings_long['condition'].dropna().unique()):
            rl = tidy.ratings_long[tidy.ratings_long['condition'] == cond]
            plot_radar_profiles(rl, pid, os.path.join(args.outdir, "figs", f"radar_{pid}_{cond.replace(' ', '_').lower()}.png"))

    # Sankey-like: planned vs rated across all tasks (aggregated across conditions)
    top_rated = (
        tidy.ratings_long[tidy.ratings_long['metric'] == 'easy_to_use']
        .groupby(['participant_id', 'task', 'anchor'])['rating'].mean().reset_index()
        .sort_values(['participant_id', 'task', 'rating'], ascending=[True, True, False])
        .groupby(['participant_id', 'task']).first().reset_index()
        .rename(columns={'anchor': 'rated_top'})
    )
    flows = tidy.rankings.merge(top_rated[['participant_id', 'task', 'rated_top']], on=['participant_id', 'task'], how='left')
    plot_sankey_planned_vs_rated(flows, os.path.join(args.outdir, "figs", "sankey_planned_vs_rated.png"))

    # Word clouds
    plot_wordclouds(tidy.comments[['condition', 'all_text']], os.path.join(args.outdir, "figs"))


if __name__ == "__main__":
    main()
