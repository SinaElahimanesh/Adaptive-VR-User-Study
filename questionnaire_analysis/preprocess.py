from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Tuple
import pandas as pd
from dateutil import parser as dtparser

from .config import ANCHORS, TASKS, COLUMN_ALIASES, ANCHOR_FIX, GLOBAL_COMMENT_KEYS
from .utils import normalize_text, find_columns_by_prefix, split_rank_order


@dataclass
class TidyData:
    participants: pd.DataFrame
    ratings_long: pd.DataFrame
    rankings: pd.DataFrame
    comments: pd.DataFrame


TASK_LABELS = {
    "key": "Key",
    "visual": "Visual",
    "controls": "Controls",
}


def _extract_demographics(df: pd.DataFrame) -> pd.DataFrame:
    norm_map = df.attrs.get("normalized_columns", {})
    rev_map = {v: k for k, v in norm_map.items()}

    def get_col(label: str) -> str:
        key = COLUMN_ALIASES.get(label, label)
        for original, norm in norm_map.items():
            if norm == key:
                return original
        # fallback: try contains
        for original, norm in norm_map.items():
            if key in norm:
                return original
        return None

    id_col = get_col("participant id")
    age_col = get_col("age")
    gender_col = get_col("gender")
    cond_col = [c for c in df.columns if normalize_text(c).startswith("trial condition")]
    cond_col = cond_col[0] if cond_col else None

    xr_col = [c for c in df.columns if normalize_text(c).startswith("experience with xr devices")]
    game_col = [c for c in df.columns if normalize_text(c).startswith("experience with videogames")]

    start_col = [c for c in df.columns if normalize_text(c) == "start time"]
    start_col = start_col[0] if start_col else None
    end_col = [c for c in df.columns if normalize_text(c) == "completion time"]
    end_col = end_col[0] if end_col else None

    participants = pd.DataFrame({
        "participant_id": df[id_col] if id_col in df.columns else pd.Series(range(1, len(df)+1), dtype="int"),
        "age": pd.to_numeric(df.get(age_col, pd.Series([None]*len(df))), errors="coerce"),
        "gender": df.get(gender_col, pd.Series([None]*len(df))).astype("string"),
        "condition": df.get(cond_col, pd.Series([None]*len(df))).astype("string"),
        "xr_experience": _coerce_scale(df.get(xr_col[0], pd.Series([None]*len(df))) if xr_col else None),
        "game_experience": _coerce_scale(df.get(game_col[0], pd.Series([None]*len(df))) if game_col else None),
        "start_time": df.get(start_col, pd.Series([None]*len(df))).astype("string"),
        "completion_time": df.get(end_col, pd.Series([None]*len(df))).astype("string"),
    })

    # Parse duration in minutes
    participants["duration_minutes"] = participants.apply(_parse_duration_row, axis=1)

    return participants


def _parse_duration_row(row: pd.Series) -> float:
    try:
        if pd.isna(row.get("start_time")) or pd.isna(row.get("completion_time")):
            return float("nan")
        start = dtparser.parse(str(row["start_time"]), dayfirst=True)
        end = dtparser.parse(str(row["completion_time"]), dayfirst=True)
        delta = (end - start).total_seconds() / 60.0
        return float(delta)
    except Exception:
        return float("nan")


def _coerce_scale(series: pd.Series | None) -> pd.Series:
    if series is None:
        return pd.Series([None])
    # Extract leading digit 1-7 if embedded in text like "7 (Very frequently)"
    return series.astype("string").str.extract(r"(\d)", expand=False).astype(float)


def _find_task_blocks(columns: List[str], task: str) -> Dict[str, List[str]]:
    # Return mapping of metric -> list of anchor columns
    # Metrics we attempt: easy_to_use, corrections
    task_norm = task.lower()
    easy_cols = []
    corr_cols = []

    for col in columns:
        n = normalize_text(col)
        if task_norm in n and "easy to use" in n:
            easy_cols.append(col)
        if task_norm in n and ("too much time correcting" in n or "correcting thing" in n or "correcting" in n):
            corr_cols.append(col)

    # We expect 4 per metric; if not, attempt to grab the next 3 columns following the first
    def expand_group(cols: List[str]) -> List[str]:
        if len(cols) >= 4:
            return cols[:4]
        if not cols:
            return []
        first_idx = columns.index(cols[0])
        return columns[first_idx:first_idx+4]

    easy_cols = expand_group(sorted(easy_cols, key=lambda c: columns.index(c)))
    corr_cols = expand_group(sorted(corr_cols, key=lambda c: columns.index(c)))

    return {"easy_to_use": easy_cols, "corrections": corr_cols}


def _anchor_from_col(col: str) -> str:
    n = normalize_text(col)
    for a in ANCHORS:
        if a.lower() in n:
            return a
    # Fallback: map numeric suffix order to anchors
    return None


def _ratings_long(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for task_key, task_label in TASK_LABELS.items():
        blocks = _find_task_blocks(list(df.columns), task_label)
        for metric, cols in blocks.items():
            if not cols:
                continue
            for i, col in enumerate(cols):
                anchor = _anchor_from_col(col) or ANCHORS[i]  # fallback to position order
                series = pd.to_numeric(df[col], errors="coerce")
                for idx, value in series.items():
                    rows.append({
                        "row_index": idx,
                        "task": task_label,
                        "metric": metric,
                        "anchor": anchor,
                        "rating": value,
                    })
    long = pd.DataFrame(rows)
    return long


def _rankings(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for task_key, task_label in TASK_LABELS.items():
        pref_cols = [c for c in df.columns if normalize_text(c).startswith(f"order the anchoring modes") and task_key in normalize_text(c)]
        if not pref_cols:
            continue
        col = pref_cols[0]
        orders = df[col].apply(split_rank_order)
        for idx, order in orders.items():
            top = order[0] if order else None
            rows.append({
                "row_index": idx,
                "task": task_label,
                "planned_order": order,
                "planned_top": top,
            })
    return pd.DataFrame(rows)


def _task_comment(df: pd.DataFrame, task_label: str) -> pd.Series:
    # Find the comment column immediately after the preference order for the task
    pref_cols = [c for c in df.columns if normalize_text(c).startswith("order the anchoring modes") and task_label.lower() in normalize_text(c)]
    if not pref_cols:
        return pd.Series([None]*len(df))
    pref_idx = df.columns.get_loc(pref_cols[0])
    # Next column is typically the explanation
    comment_col = df.columns[pref_idx + 1] if pref_idx + 1 < len(df.columns) else None
    if comment_col is None:
        return pd.Series([None]*len(df))
    return df[comment_col]


def _comments(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for idx in range(len(df)):
        entry = {"row_index": idx}
        for t in TASKS:
            entry[f"comment_{t.lower()}"] = str(_task_comment(df, t).iloc[idx]) if len(df) > idx else None
        # Global comments
        for key in GLOBAL_COMMENT_KEYS:
            cols = [c for c in df.columns if normalize_text(c) == normalize_text(key)]
            if cols:
                entry[normalize_text(key)] = str(df[cols[0]].iloc[idx])
        rows.append(entry)
    com = pd.DataFrame(rows)
    # Combined text per row
    text_cols = [c for c in com.columns if c != "row_index"]
    com["all_text"] = com[text_cols].astype(str).agg(". ".join, axis=1)
    return com


def build_tidy_frames(raw: pd.DataFrame) -> TidyData:
    participants = _extract_demographics(raw)
    long = _ratings_long(raw)
    rankings = _rankings(raw)
    comments = _comments(raw)

    # Attach participant info by row_index alignment (assume 1 row per participant)
    long = long.merge(participants.reset_index().rename(columns={"index": "row_index"})[["row_index", "participant_id", "condition"]], on="row_index", how="left")
    rankings = rankings.merge(participants.reset_index().rename(columns={"index": "row_index"})[["row_index", "participant_id", "condition"]], on="row_index", how="left")
    comments = comments.merge(participants.reset_index().rename(columns={"index": "row_index"})[["row_index", "participant_id", "condition"]], on="row_index", how="left")

    return TidyData(
        participants=participants,
        ratings_long=long,
        rankings=rankings,
        comments=comments,
    )
