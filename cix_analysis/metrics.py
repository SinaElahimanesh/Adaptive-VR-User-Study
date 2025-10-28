from __future__ import annotations

from typing import Dict
import numpy as np
import pandas as pd

from .config import TASKS, ANCHORS


def compute_preference_strength_index(ratings_long: pd.DataFrame, metric: str = "easy_to_use") -> pd.DataFrame:
    df = ratings_long[ratings_long['metric'] == metric]
    rows = []
    for (pid, cond, task), sub in df.groupby(['participant_id', 'condition', 'task']):
        scores = sub.set_index('anchor')['rating']
        scores = scores.reindex(ANCHORS)
        if scores.isna().all():
            continue
        sorted_scores = scores.dropna().sort_values(ascending=False)
        if len(sorted_scores) == 0:
            continue
        top = sorted_scores.iloc[0]
        second = sorted_scores.iloc[1] if len(sorted_scores) > 1 else np.nan
        psi = top - second if np.isfinite(top) and np.isfinite(second) else np.nan
        rows.append({
            "participant_id": pid,
            "condition": cond,
            "task": task,
            "psi": psi,
            "top_anchor": sorted_scores.index[0]
        })
    return pd.DataFrame(rows)


def compute_anchor_adaptability(ratings_long: pd.DataFrame, metric: str = "easy_to_use") -> pd.DataFrame:
    df = ratings_long[ratings_long['metric'] == metric]
    top_per = (
        df.groupby(['participant_id', 'condition', 'task', 'anchor'])['rating'].mean().reset_index()
        .sort_values(['participant_id', 'condition', 'task', 'rating'], ascending=[True, True, True, False])
        .groupby(['participant_id', 'condition', 'task']).first().reset_index()
    )
    rows = []
    for (pid, cond), sub in top_per.groupby(['participant_id', 'condition']):
        anchors = sub.sort_values('task')['anchor'].tolist()
        changes = sum(1 for i in range(1, len(anchors)) if anchors[i] != anchors[i-1])
        rows.append({"participant_id": pid, "condition": cond, "adaptability_changes": changes})
    return pd.DataFrame(rows)


    


def summarize_adaptability_within(adaptability_df: pd.DataFrame) -> pd.DataFrame:
    if adaptability_df.empty:
        return pd.DataFrame()
    return (adaptability_df.groupby('condition')['adaptability_changes']
            .agg(['mean', 'median', 'std', 'count']).reset_index()
            .rename(columns={'mean': 'mean_changes', 'median': 'median_changes', 'std': 'std_changes'}))


def summarize_adaptability_across(across_df: pd.DataFrame, col_name: str) -> pd.DataFrame:
    if across_df.empty:
        return pd.DataFrame()
    # across_df expected columns: participant_id, task, <col_name>
    return (across_df.groupby('task')[col_name]
            .agg(['mean', 'median', 'std', 'count']).reset_index()
            .rename(columns={'mean': 'mean_changes', 'median': 'median_changes', 'std': 'std_changes'}))
