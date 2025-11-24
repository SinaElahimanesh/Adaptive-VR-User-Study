from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import pandas as pd

# Ensure local package import when running directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from logs_analysis.io import discover_participants
from logs_analysis.metrics import (
    build_tidy_ui_events,
    summarize_ui_tasks,
    summarize_motion,
    inter_completion_intervals,
    inter_completion_intervals_by_task,
    summarize_ui_tasks_by_anchor,
    compute_anchor_significance,
)
from logs_analysis.mixed import (
    load_demographics,
    prepare_performance_tables,
    analyze_performance_vs_demographics,
)
# Defer importing plotting utilities until needed to avoid heavy deps in headless runs


def save_table(df: pd.DataFrame, path: str) -> None:
    os.makedirs(Path(path).parent, exist_ok=True)
    df.to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze StudyLogs")
    parser.add_argument("--logs_root", default="StudyLogs", help="Path to StudyLogs root directory")
    parser.add_argument("--outdir", default="outputs/logs", help="Directory to write log-derived outputs")
    parser.add_argument("--questionnaire", default="Questionnaire.csv", help="Path to Questionnaire.csv for demographics (optional)")
    parser.add_argument("--skip_figs", action="store_true", help="Skip generating figures (headless)")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # Tidy UI events across all participants/conditions
    tidy_ui = build_tidy_ui_events(args.logs_root)
    save_table(tidy_ui, os.path.join(args.outdir, "tidy_ui_events.csv"))

    # UI task summaries
    ui_summary = summarize_ui_tasks(tidy_ui)
    save_table(ui_summary, os.path.join(args.outdir, "ui_task_summary.csv"))
    # UI task summaries by anchor (coordinate system)
    ui_summary_anchor = summarize_ui_tasks_by_anchor(tidy_ui)
    save_table(ui_summary_anchor, os.path.join(args.outdir, "ui_task_summary_by_anchor.csv"))
    # Overall roll-up across participants per condition×task×anchor
    if not ui_summary_anchor.empty:
        overall_anchor = (
            ui_summary_anchor
            .groupby(['condition', 'task', 'anchor'])
            .agg(
                participants=('participant', 'nunique'),
                completions=('num_completions', 'sum'),
                mean_duration_s=('mean_duration_s', 'mean'),
                median_duration_s=('median_duration_s', 'mean'),
                std_duration_s=('std_duration_s', 'mean'),
                accuracy=('accuracy', 'mean'),
            )
            .reset_index()
            .sort_values(['condition', 'task', 'anchor'])
        )
        save_table(overall_anchor, os.path.join(args.outdir, "ui_task_summary_by_anchor_overall.csv"))
        # Significance tests across anchors within each condition×task
        omni, pair = compute_anchor_significance(ui_summary_anchor)
        if not omni.empty:
            save_table(omni, os.path.join(args.outdir, "ui_anchor_significance.csv"))
        if not pair.empty:
            save_table(pair, os.path.join(args.outdir, "ui_anchor_significance_pairwise.csv"))

    # Motion capture summaries (lightweight)
    motion_summary = summarize_motion(args.logs_root)
    save_table(motion_summary, os.path.join(args.outdir, "motion_summary.csv"))

    # Additional metrics
    ic = inter_completion_intervals(tidy_ui)
    save_table(ic['intervals'], os.path.join(args.outdir, "inter_completion_intervals.csv"))
    save_table(ic['summary'], os.path.join(args.outdir, "inter_completion_summary.csv"))
    # Task-level intervals
    ic_task = inter_completion_intervals_by_task(tidy_ui)
    save_table(ic_task['intervals'], os.path.join(args.outdir, "inter_completion_intervals_by_task.csv"))
    save_table(ic_task['summary'], os.path.join(args.outdir, "inter_completion_summary_by_task.csv"))

    # Figures
    fig_paths = []
    if not args.skip_figs:
        # Import plotting functions lazily to avoid importing matplotlib/seaborn when skipping
        from logs_analysis.visuals import (
            plot_duration_boxplots,
            plot_accuracy_bars,
            plot_num_completions,
            plot_mean_duration_heatmap,
            plot_motion_duration,
            plot_perf_scatter_intervals,
        )
        figs_dir = os.path.join(args.outdir, 'figs')
        os.makedirs(figs_dir, exist_ok=True)
        for p in [
            plot_duration_boxplots(tidy_ui, figs_dir),
            plot_accuracy_bars(ui_summary, figs_dir),
            plot_num_completions(ui_summary, figs_dir),
            plot_mean_duration_heatmap(ui_summary, figs_dir),
            plot_motion_duration(motion_summary, figs_dir),
        ]:
            if p:
                fig_paths.append(p)

    # Optional: performance vs demographics (requires Questionnaire.csv)
    qpath = Path(args.questionnaire)
    if qpath.exists():
        try:
            demo = load_demographics(str(qpath))
            perf = prepare_performance_tables(
                inter_completion_summary_csv=os.path.join(args.outdir, "inter_completion_summary.csv"),
                ui_task_summary_csv=os.path.join(args.outdir, "ui_task_summary.csv"),
                inter_completion_summary_by_task_csv=os.path.join(args.outdir, "inter_completion_summary_by_task.csv"),
            )
            results = analyze_performance_vs_demographics(
                demographics=demo,
                intervals=perf["intervals"],
                durations_overall=perf["durations_overall"],
                intervals_by_task=perf.get("intervals_by_task"),
                durations_by_task=perf.get("durations_by_task", perf["durations_by_task"]),
            )
            # Save stats
            save_table(results["corr_intervals"], os.path.join(args.outdir, "performance_correlations_intervals.csv"))
            save_table(results["corr_durations"], os.path.join(args.outdir, "performance_correlations_durations.csv"))
            save_table(results["group_intervals"], os.path.join(args.outdir, "performance_group_tests_intervals.csv"))
            save_table(results["group_durations"], os.path.join(args.outdir, "performance_group_tests_durations.csv"))
            # By-task stats (if available)
            if "corr_intervals_by_task" in results:
                save_table(results["corr_intervals_by_task"], os.path.join(args.outdir, "performance_correlations_intervals_by_task.csv"))
            if "group_intervals_by_task" in results:
                save_table(results["group_intervals_by_task"], os.path.join(args.outdir, "performance_group_tests_intervals_by_task.csv"))
            if "corr_durations_by_task" in results:
                save_table(results["corr_durations_by_task"], os.path.join(args.outdir, "performance_correlations_durations_by_task.csv"))
            if "group_durations_by_task" in results:
                save_table(results["group_durations_by_task"], os.path.join(args.outdir, "performance_group_tests_durations_by_task.csv"))
            # Figures: scatter for intervals vs each predictor
            if not args.skip_figs:
                joined = results["joined_intervals"]
                # Import inside block
                from logs_analysis.visuals import plot_perf_scatter_intervals
                figs_dir = os.path.join(args.outdir, 'figs')
                os.makedirs(figs_dir, exist_ok=True)
                for demo_col in ["age", "xr_experience", "videogame_experience"]:
                    fig = plot_perf_scatter_intervals(joined, demo_col, figs_dir)
                    if fig:
                        fig_paths.append(fig)
        except Exception as e:
            print(f"[WARN] Performance-vs-demographics analysis skipped due to error: {e}")
    else:
        print(f"[INFO] Questionnaire not found at {qpath}; skipping performance-vs-demographics.")

    print(f"Saved logs analyses to {args.outdir}")
    print("Participants detected:", ", ".join(discover_participants(args.logs_root)))
    if not ui_summary.empty:
        print("\nUI task summary (head):\n" + ui_summary.head(10).to_string(index=False))
    if not ic['summary'].empty:
        print("\nInter-completion intervals (head):\n" + ic['summary'].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
