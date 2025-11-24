from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from scipy.stats import kruskal, mannwhitneyu

from .io import discover_participants, load_ui_log, scan_motion_log


CONDITIONS = ["Stationary", "SemiStationary", "Moving"]
UI_TASK_MAP = {
    "VisualTask": "Visual",
    "KeyTask": "Key",
    "ControlTask": "Controls",
}


def build_tidy_ui_events(logs_root: str) -> pd.DataFrame:
    rows = []
    for p in discover_participants(logs_root):
        pdir = str(Path(logs_root) / p)
        for cond in CONDITIONS:
            df = load_ui_log(pdir, cond)
            if df.empty:
                continue
            for _, r in df.iterrows():
                rows.append({
                    "participant": p,
                    "condition": cond,
                    "task": UI_TASK_MAP.get(str(r.get('UIName')), str(r.get('UIName'))),
                    "event_type": r.get('EventType'),
                    "coordinate_system": r.get('CoordinateSystem'),
                    "correct": r.get('Correct'),
                    "timestamp": r.get('Timestamp'),
                    "duration": r.get('Duration'),
                })
    return pd.DataFrame(rows)


def summarize_ui_tasks(tidy_ui: pd.DataFrame) -> pd.DataFrame:
    if tidy_ui.empty:
        return pd.DataFrame()
    tc = tidy_ui[tidy_ui['event_type'] == 'TaskCompletion'].copy()
    tc['is_correct'] = tc['correct'].astype('boolean')
    summary = (tc.groupby(['participant', 'condition', 'task'])
               .agg(num_completions=('event_type', 'count'),
                    mean_duration_s=('duration', 'mean'),
                    median_duration_s=('duration', 'median'),
                    std_duration_s=('duration', 'std'),
                    accuracy=('is_correct', 'mean'))
               .reset_index())
    return summary


def summarize_ui_tasks_by_anchor(tidy_ui: pd.DataFrame) -> pd.DataFrame:
    """Summarize TaskCompletion performance by participant×condition×task×anchor.
    Anchor is taken from the 'coordinate_system' field in the tidy UI events.
    """
    if tidy_ui.empty:
        return pd.DataFrame()
    tc = tidy_ui[tidy_ui['event_type'] == 'TaskCompletion'].copy()
    # Normalize anchor names for consistency and map limb variants to canonical anchors
    raw_anchor = tc['coordinate_system'].astype('string').str.strip().str.lower()
    anchor_norm = (
        raw_anchor
        .mask(raw_anchor.str.contains('limb') | raw_anchor.str.contains('arm'), 'arm')
        .mask(raw_anchor.str.contains('head'), 'head')
        .mask(raw_anchor.str.contains('torso'), 'torso')
        .mask(raw_anchor.str.contains('world'), 'world')
        .fillna(raw_anchor)
    )
    tc['anchor'] = anchor_norm.str.title()
    tc['is_correct'] = tc['correct'].astype('boolean')
    summary = (tc.groupby(['participant', 'condition', 'task', 'anchor'])
               .agg(num_completions=('event_type', 'count'),
                    mean_duration_s=('duration', 'mean'),
                    median_duration_s=('duration', 'median'),
                    std_duration_s=('duration', 'std'),
                    accuracy=('is_correct', 'mean'))
               .reset_index())
    return summary


    


def summarize_motion(logs_root: str) -> pd.DataFrame:
    rows = []
    for p in discover_participants(logs_root):
        pdir = str(Path(logs_root) / p)
        for cond in CONDITIONS:
            info = scan_motion_log(pdir, cond)
            if not info:
                continue
            rows.append({
                'participant': p,
                'condition': cond,
                **info
            })
    return pd.DataFrame(rows)


# Additional metrics

def inter_completion_intervals(tidy_ui: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Compute intervals (s) between consecutive TaskCompletion events.
    Returns raw intervals and per participant×condition summaries.
    """
    if tidy_ui.empty:
        return {"intervals": pd.DataFrame(), "summary": pd.DataFrame()}
    tc = tidy_ui[tidy_ui['event_type'] == 'TaskCompletion'][['participant', 'condition', 'timestamp']].dropna()
    tc = tc.sort_values(['participant', 'condition', 'timestamp'])
    # Compute diffs per participant×condition
    tc['interval_s'] = tc.groupby(['participant', 'condition'])['timestamp'].diff()
    intervals = tc.dropna(subset=['interval_s']).copy()
    summary = (intervals.groupby(['participant', 'condition'])['interval_s']
               .agg(['mean', 'median', 'std', 'count']).reset_index()
               .rename(columns={'mean': 'mean_interval_s', 'median': 'median_interval_s', 'std': 'std_interval_s'}))
    return {"intervals": intervals, "summary": summary}


def inter_completion_intervals_by_task(tidy_ui: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Compute intervals (s) between consecutive TaskCompletion events per task.
    Returns:
      - intervals: participant×condition×task rows with interval_s
      - summary: per participant×condition×task aggregated stats
    """
    if tidy_ui.empty:
        return {"intervals": pd.DataFrame(), "summary": pd.DataFrame()}
    cols = ['participant', 'condition', 'task', 'timestamp']
    tc = tidy_ui[tidy_ui['event_type'] == 'TaskCompletion'][cols].dropna(subset=['timestamp'])
    if tc.empty:
        return {"intervals": pd.DataFrame(), "summary": pd.DataFrame()}
    tc = tc.sort_values(['participant', 'condition', 'task', 'timestamp'])
    tc['interval_s'] = tc.groupby(['participant', 'condition', 'task'])['timestamp'].diff()
    intervals = tc.dropna(subset=['interval_s']).copy()
    summary = (intervals.groupby(['participant', 'condition', 'task'])['interval_s']
               .agg(['mean', 'median', 'std', 'count']).reset_index()
               .rename(columns={'mean': 'mean_interval_s', 'median': 'median_interval_s', 'std': 'std_interval_s'}))
    return {"intervals": intervals, "summary": summary}


    


    

# Significance tests for anchor effects within each condition×task
def compute_anchor_significance(
    ui_summary_anchor: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    For each (condition, task), test if completion times differ across anchors.
    - Uses participant-level mean_duration_s from ui_summary_anchor.
    - Omnibus: Kruskal-Wallis across available anchors (nonparametric, unequal n).
    - Pairwise: Mann-Whitney U for all anchor pairs with Holm correction.
    Returns:
      (omnibus_df, pairwise_df)
    """
    if ui_summary_anchor.empty:
        return pd.DataFrame(), pd.DataFrame()

    required_cols = {'condition', 'task', 'anchor', 'participant', 'mean_duration_s'}
    if not required_cols.issubset(set(ui_summary_anchor.columns)):
        return pd.DataFrame(), pd.DataFrame()

    omnibus_rows: List[Dict] = []
    pairwise_rows: List[Dict] = []

    # Prepare groups
    grouped = ui_summary_anchor.groupby(['condition', 'task'])
    for (cond, task), df_ct in grouped:
        anchor_groups = []
        labels = []
        for anc, df_a in df_ct.groupby('anchor'):
            vals = df_a['mean_duration_s'].dropna().values
            if len(vals) >= 2:
                anchor_groups.append(vals)
                labels.append(anc)
        k = len(anchor_groups)
        n_total = int(sum(len(g) for g in anchor_groups))
        if k < 2:
            continue
        try:
            H, p = kruskal(*anchor_groups)
        except Exception:
            continue
        # Epsilon-squared effect size for Kruskal
        eps2 = (H - (k - 1)) / (n_total - 1) if n_total > 1 else np.nan
        omnibus_rows.append({
            'condition': cond,
            'task': task,
            'anchors': k,
            'n_total': n_total,
            'H_kruskal': H,
            'p_kruskal': p,
            'epsilon_squared': eps2,
        })

        # Pairwise Mann-Whitney with Holm correction
        raw_tests: List[Tuple[str, str, float, float, int, int]] = []
        for i in range(k):
            for j in range(i + 1, k):
                a_i, a_j = labels[i], labels[j]
                x, y = anchor_groups[i], anchor_groups[j]
                try:
                    u_stat, p_pair = mannwhitneyu(x, y, alternative='two-sided')
                except Exception:
                    continue
                raw_tests.append((a_i, a_j, u_stat, p_pair, len(x), len(y)))
        # Holm adjustment
        if raw_tests:
            m = len(raw_tests)
            # sort by p ascending
            raw_tests_sorted = sorted(raw_tests, key=lambda t: t[3])
            adjusted: Dict[Tuple[str, str], float] = {}
            for idx, (a_i, a_j, _u, p_raw, _ni, _nj) in enumerate(raw_tests_sorted, start=1):
                p_holm = min((m - idx + 1) * p_raw, 1.0)
                adjusted[(a_i, a_j)] = p_holm
            for a_i, a_j, u_stat, p_raw, ni, nj in raw_tests:
                pairwise_rows.append({
                    'condition': cond,
                    'task': task,
                    'anchor_a': a_i,
                    'anchor_b': a_j,
                    'n_a': ni,
                    'n_b': nj,
                    'U_mannwhitney': u_stat,
                    'p_raw': p_raw,
                    'p_holm': adjusted.get((a_i, a_j), np.nan),
                })

    return pd.DataFrame(omnibus_rows), pd.DataFrame(pairwise_rows)
