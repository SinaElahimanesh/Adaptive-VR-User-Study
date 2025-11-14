from __future__ import annotations

import os
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set(style="whitegrid")


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def plot_duration_boxplots(tidy_ui: pd.DataFrame, outdir: str) -> str:
    _ensure_dir(outdir)
    tc = tidy_ui[tidy_ui['event_type'] == 'TaskCompletion'].dropna(subset=['duration'])
    if tc.empty:
        return ""
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=tc, x='task', y='duration', hue='condition')
    plt.ylabel('Completion time (s)')
    plt.xlabel('Task')
    plt.title('Task completion time distributions by condition')
    plt.legend(title='Condition', bbox_to_anchor=(1.05, 1), loc='upper left')
    fname = os.path.join(outdir, 'boxplot_duration_by_task_condition.png')
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close()
    return fname


def plot_accuracy_bars(ui_summary: pd.DataFrame, outdir: str) -> str:
    _ensure_dir(outdir)
    if ui_summary.empty:
        return ""
    agg = (ui_summary.groupby(['condition', 'task'])['accuracy']
           .mean().reset_index())
    plt.figure(figsize=(7, 4))
    sns.barplot(data=agg, x='task', y='accuracy', hue='condition')
    plt.ylim(0, 1)
    plt.ylabel('Accuracy')
    plt.title('Accuracy by condition and task')
    plt.legend(title='Condition', bbox_to_anchor=(1.05, 1), loc='upper left')
    fname = os.path.join(outdir, 'bar_accuracy_by_task_condition.png')
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close()
    return fname


def plot_num_completions(ui_summary: pd.DataFrame, outdir: str) -> str:
    _ensure_dir(outdir)
    if ui_summary.empty:
        return ""
    agg = (ui_summary.groupby(['condition', 'task'])['num_completions']
           .median().reset_index())
    plt.figure(figsize=(7, 4))
    sns.barplot(data=agg, x='task', y='num_completions', hue='condition')
    plt.ylabel('Median completions per participant')
    plt.title('Completions by condition and task')
    plt.legend(title='Condition', bbox_to_anchor=(1.05, 1), loc='upper left')
    fname = os.path.join(outdir, 'bar_completions_by_task_condition.png')
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close()
    return fname


def plot_mean_duration_heatmap(ui_summary: pd.DataFrame, outdir: str) -> str:
    _ensure_dir(outdir)
    if ui_summary.empty:
        return ""
    agg = (ui_summary.groupby(['condition', 'task'])['mean_duration_s']
           .mean().reset_index())
    pivot = agg.pivot(index='condition', columns='task', values='mean_duration_s')
    plt.figure(figsize=(6, 4))
    sns.heatmap(pivot, annot=True, fmt='.1f', cmap='mako')
    plt.title('Mean completion time (s)')
    fname = os.path.join(outdir, 'heatmap_mean_duration.png')
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close()
    return fname


def plot_motion_duration(motion_summary: pd.DataFrame, outdir: str) -> str:
    _ensure_dir(outdir)
    if motion_summary.empty:
        return ""
    agg = (motion_summary.groupby('condition')['duration_seconds']
           .mean().reset_index())
    plt.figure(figsize=(6, 4))
    sns.barplot(data=agg, x='condition', y='duration_seconds')
    plt.ylabel('Avg recording duration (s)')
    plt.title('Motion recording duration by condition')
    fname = os.path.join(outdir, 'bar_motion_duration_by_condition.png')
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close()
    return fname


def plot_perf_scatter_intervals(perf_intervals_joined: pd.DataFrame, demo_col: str, outdir: str) -> str:
    """Scatter of mean_interval_s vs a demographic predictor, colored by condition."""
    _ensure_dir(outdir)
    df = perf_intervals_joined.copy()
    if df.empty or demo_col not in df.columns:
        return ""
    # Filter rows with both values present
    df = df[['mean_interval_s', 'condition', demo_col]].copy()
    df = df.dropna(subset=['mean_interval_s', demo_col])
    if df.empty:
        return ""
    plt.figure(figsize=(7, 5))
    sns.scatterplot(data=df, x=demo_col, y='mean_interval_s', hue='condition')
    sns.regplot(
        data=df,
        x=demo_col,
        y='mean_interval_s',
        scatter=False,
        ci=None,
        color='black',
        line_kws={'linewidth': 1, 'alpha': 0.6},
    )
    plt.xlabel(demo_col.replace('_', ' ').title())
    plt.ylabel('Mean inter-completion interval (s)')
    plt.title(f'Performance vs {demo_col.replace("_", " ").title()}')
    plt.legend(title='Condition', bbox_to_anchor=(1.05, 1), loc='upper left')
    fname = os.path.join(outdir, f'scatter_intervals_vs_{demo_col}.png')
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close()
    return fname
