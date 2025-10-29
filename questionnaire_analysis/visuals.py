from __future__ import annotations

import os
from typing import Dict, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from wordcloud import WordCloud

sns.set(style="whitegrid")


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def plot_preference_barcharts(pref_counts: pd.DataFrame, outdir: str) -> List[str]:
    _ensure_dir(outdir)
    saved = []
    for task, sub in pref_counts.groupby('task'):
        plt.figure(figsize=(6, 4))
        sns.barplot(data=sub, x='planned_top', y='proportion', hue='condition')
        plt.title(f"Planned top anchors for {task}")
        plt.ylabel("Proportion")
        plt.xlabel("Anchor")
        plt.ylim(0, 1)
        plt.legend(title="Condition", bbox_to_anchor=(1.05, 1), loc='upper left')
        fname = os.path.join(outdir, f"pref_barchart_{task.lower()}.png")
        plt.tight_layout()
        plt.savefig(fname, dpi=150)
        plt.close()
        saved.append(fname)
    return saved


def plot_heatmaps(desc_by_cond: pd.DataFrame, outdir: str) -> List[str]:
    _ensure_dir(outdir)
    saved = []
    for task, sub in desc_by_cond.groupby('task'):
        pivot = sub.pivot_table(index='condition', columns='anchor', values='mean')
        plt.figure(figsize=(6, 4))
        sns.heatmap(pivot, annot=True, fmt=".2f", cmap="viridis")
        plt.title(f"Average ease ratings by condition and anchor - {task}")
        fname = os.path.join(outdir, f"heatmap_{task.lower()}.png")
        plt.tight_layout()
        plt.savefig(fname, dpi=150)
        plt.close()
        saved.append(fname)
    return saved


def plot_radar_profiles(ratings_long: pd.DataFrame, participant_id: int, outpath: str) -> str:
    # Single participant radar over anchors for each task
    import numpy as np
    tasks = sorted(ratings_long['task'].dropna().unique())
    anchors = sorted(ratings_long['anchor'].dropna().unique())
    N = len(anchors)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, axes = plt.subplots(1, len(tasks), subplot_kw=dict(polar=True), figsize=(4*len(tasks), 4))
    if len(tasks) == 1:
        axes = [axes]
    for ax, task in zip(axes, tasks):
        sub = ratings_long[(ratings_long['participant_id'] == participant_id) & (ratings_long['task'] == task) & (ratings_long['metric'] == 'easy_to_use')]
        means = sub.groupby('anchor')['rating'].mean().reindex(anchors).fillna(0).tolist()
        means += means[:1]
        ax.plot(angles, means, linewidth=2)
        ax.fill(angles, means, alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(anchors)
        ax.set_title(task)
        ax.set_ylim(0, 7)
    plt.suptitle(f"Participant {participant_id} - Anchor preferences (ease)")
    _ensure_dir(os.path.dirname(outpath))
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()
    return outpath


def plot_sankey_planned_vs_rated(flows: pd.DataFrame, outpath: str) -> str:
    # Use matplotlib stacked bars as a simple sankey-like proxy if plotly is not desired
    import matplotlib.pyplot as plt
    import seaborn as sns
    _ensure_dir(os.path.dirname(outpath))
    plt.figure(figsize=(6, 4))
    sns.countplot(data=flows, x='planned_top', hue='rated_top')
    plt.title("Planned vs Rated Top Anchor (All tasks)")
    plt.xlabel("Planned top")
    plt.ylabel("Count")
    plt.legend(title="Rated top", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()
    return outpath


def plot_wordclouds(comments_by_condition: pd.DataFrame, outdir: str) -> List[str]:
    _ensure_dir(outdir)
    saved = []
    for cond, sub in comments_by_condition.groupby('condition'):
        text = " ".join(str(t) for t in sub['all_text'] if isinstance(t, str))
        if not text.strip():
            continue
        wc = WordCloud(width=800, height=400, background_color="white").generate(text)
        plt.figure(figsize=(8, 4))
        plt.imshow(wc, interpolation='bilinear')
        plt.axis('off')
        plt.title(f"Word Cloud - {cond}")
        fname = os.path.join(outdir, f"wordcloud_{str(cond).lower().replace(' ', '_')}.png")
        plt.tight_layout()
        plt.savefig(fname, dpi=150)
        plt.close()
        saved.append(fname)
    return saved
