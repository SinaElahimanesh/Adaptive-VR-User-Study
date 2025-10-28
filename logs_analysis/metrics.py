from __future__ import annotations

from pathlib import Path
from typing import Dict, List
import numpy as np
import pandas as pd

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


    


    
