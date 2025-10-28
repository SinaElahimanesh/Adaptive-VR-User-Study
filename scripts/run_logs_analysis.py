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
)
from logs_analysis.visuals import (
    plot_duration_boxplots,
    plot_accuracy_bars,
    plot_num_completions,
    plot_mean_duration_heatmap,
    plot_motion_duration,
)


def save_table(df: pd.DataFrame, path: str) -> None:
    os.makedirs(Path(path).parent, exist_ok=True)
    df.to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze StudyLogs")
    parser.add_argument("--logs_root", default="StudyLogs", help="Path to StudyLogs root directory")
    parser.add_argument("--outdir", default="outputs/logs", help="Directory to write log-derived outputs")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # Tidy UI events across all participants/conditions
    tidy_ui = build_tidy_ui_events(args.logs_root)
    save_table(tidy_ui, os.path.join(args.outdir, "tidy_ui_events.csv"))

    # UI task summaries
    ui_summary = summarize_ui_tasks(tidy_ui)
    save_table(ui_summary, os.path.join(args.outdir, "ui_task_summary.csv"))

    # Motion capture summaries (lightweight)
    motion_summary = summarize_motion(args.logs_root)
    save_table(motion_summary, os.path.join(args.outdir, "motion_summary.csv"))

    # Additional metrics
    ic = inter_completion_intervals(tidy_ui)
    save_table(ic['intervals'], os.path.join(args.outdir, "inter_completion_intervals.csv"))
    save_table(ic['summary'], os.path.join(args.outdir, "inter_completion_summary.csv"))

    # Figures
    figs_dir = os.path.join(args.outdir, 'figs')
    os.makedirs(figs_dir, exist_ok=True)
    fig_paths = []
    for p in [
        plot_duration_boxplots(tidy_ui, figs_dir),
        plot_accuracy_bars(ui_summary, figs_dir),
        plot_num_completions(ui_summary, figs_dir),
        plot_mean_duration_heatmap(ui_summary, figs_dir),
        plot_motion_duration(motion_summary, figs_dir),
    ]:
        if p:
            fig_paths.append(p)

    print(f"Saved logs analyses to {args.outdir}")
    print("Participants detected:", ", ".join(discover_participants(args.logs_root)))
    if not ui_summary.empty:
        print("\nUI task summary (head):\n" + ui_summary.head(10).to_string(index=False))
    if not ic['summary'].empty:
        print("\nInter-completion intervals (head):\n" + ic['summary'].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
