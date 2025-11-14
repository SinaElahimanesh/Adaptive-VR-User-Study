from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats


def _read_questionnaire_csv_robust(path: str) -> pd.DataFrame:
    """Try multiple encodings and normalize NBSP in headers."""
    last_err: Optional[Exception] = None
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            df = pd.read_csv(path, sep=";", engine="python", encoding=enc)
            # Normalize NBSP in headers
            df.columns = [c.replace("\xa0", " ") for c in df.columns]
            return df
        except Exception as e:
            last_err = e
            continue
    raise last_err if last_err else RuntimeError("Failed to read questionnaire CSV")


def _parse_leading_int(value: object) -> Optional[int]:
    """Extract the leading integer in a cell like '7 (Very frequently)'; returns None if not found."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    s = str(value).strip()
    num = ""
    for ch in s:
        if ch.isdigit():
            num += ch
        else:
            break
    if num == "":
        return None
    try:
        return int(num)
    except ValueError:
        return None


def load_demographics(questionnaire_csv: str) -> pd.DataFrame:
    """Load demographics from Questionnaire.csv and normalize key columns."""
    df = _read_questionnaire_csv_robust(questionnaire_csv)
    # Normalize header names
    cols = {c: c.strip() for c in df.columns}
    df.rename(columns=cols, inplace=True)
    # Standard fields we care about
    pid_col = "Participant ID"
    age_col = "Age"
    xr_col = "Answer the following statements, from a scale of 1 to 7:.Experience with XR devices (VR/AR/MR)"
    vg_col = "Answer the following statements, from a scale of 1 to 7:.Experience with videogames"
    if pid_col not in df.columns:
        # fallback to a more generic guess
        for c in df.columns:
            if "Participant ID" in c:
                pid_col = c
                break
    demo = pd.DataFrame({
        "participant_id": pd.to_numeric(df[pid_col], errors="coerce"),
        "age": pd.to_numeric(df.get(age_col), errors="coerce"),
        "xr_experience": df.get(xr_col).apply(_parse_leading_int) if xr_col in df.columns else None,
        "videogame_experience": df.get(vg_col).apply(_parse_leading_int) if vg_col in df.columns else None,
    })
    demo = demo.dropna(subset=["participant_id"]).copy()
    demo["participant_id"] = demo["participant_id"].astype(int)
    return demo


def _participant_str_to_id(participant: str) -> Optional[int]:
    """Convert 'Participant_12' -> 12."""
    try:
        return int(str(participant).split("_")[-1])
    except Exception:
        return None


def prepare_performance_tables(
    inter_completion_summary_csv: str,
    ui_task_summary_csv: str,
    inter_completion_summary_by_task_csv: Optional[str] = None,
) -> Dict[str, pd.DataFrame]:
    """Load performance metrics computed from StudyLogs outputs.
    Returns:
      - intervals: mean_interval_s per participant×condition
      - durations_by_task: mean_duration_s per participant×condition×task
      - durations_overall: mean over tasks per participant×condition
    """
    intervals = pd.read_csv(inter_completion_summary_csv)
    durations = pd.read_csv(ui_task_summary_csv)
    intervals_by_task = pd.DataFrame()
    if inter_completion_summary_by_task_csv and Path(inter_completion_summary_by_task_csv).exists():
        intervals_by_task = pd.read_csv(inter_completion_summary_by_task_csv)
    # Map Participant_# -> int id
    intervals["participant_id"] = intervals["participant"].apply(_participant_str_to_id)
    durations["participant_id"] = durations["participant"].apply(_participant_str_to_id)
    if not intervals_by_task.empty:
        intervals_by_task["participant_id"] = intervals_by_task["participant"].apply(_participant_str_to_id)
    intervals = intervals.dropna(subset=["participant_id"]).copy()
    durations = durations.dropna(subset=["participant_id"]).copy()
    # Aggregate durations across tasks (overall per participant×condition)
    durations_overall = (
        durations.groupby(["participant_id", "participant", "condition"])["mean_duration_s"]
        .mean()
        .reset_index()
        .rename(columns={"mean_duration_s": "mean_duration_overall_s"})
    )
    return {
        "intervals": intervals,
        "durations_by_task": durations,
        "durations_overall": durations_overall,
        "intervals_by_task": intervals_by_task,
    }


def _spearman_for_groups(
    df: pd.DataFrame,
    value_col: str,
    predictors: List[str],
    group_cols: List[str],
) -> pd.DataFrame:
    """Compute Spearman ρ and p per group for the given predictors."""
    rows: List[Dict[str, object]] = []
    if not group_cols:
        groups = [((), df)]
    else:
        groups = df.groupby(group_cols, dropna=False)
    for group_key, sub in groups:
        for pred in predictors:
            x = pd.to_numeric(sub[pred], errors="coerce")
            y = pd.to_numeric(sub[value_col], errors="coerce")
            ok = x.notna() & y.notna()
            if ok.sum() >= 3:
                rho, p = stats.spearmanr(x[ok], y[ok])
            else:
                rho, p = np.nan, np.nan
            row: Dict[str, object] = {**{gc: g for gc, g in zip(group_cols, group_key if isinstance(group_key, tuple) else (group_key,))}}
            row.update({
                "metric": value_col,
                "predictor": pred,
                "rho": rho,
                "p_value": p,
                "n": int(ok.sum()),
            })
            rows.append(row)
    return pd.DataFrame(rows)


def _kruskal_by_bins(
    df: pd.DataFrame,
    value_col: str,
    bin_specs: Dict[str, Tuple[pd.Series, List[str]]],
    group_cols: List[str],
) -> pd.DataFrame:
    """Kruskal–Wallis tests across predefined bins per predictor."""
    rows: List[Dict[str, object]] = []
    if not group_cols:
        groups = [((), df)]
    else:
        groups = df.groupby(group_cols, dropna=False)
    for group_key, sub in groups:
        for pred, (labels, order) in bin_specs.items():
            sub2 = sub[[pred, value_col]].copy()
            sub2["bin"] = labels
            # Collect samples per bin
            samples: List[pd.Series] = []
            bin_ns: Dict[str, int] = {}
            for label in order:
                vals = pd.to_numeric(sub2.loc[sub2["bin"] == label, value_col], errors="coerce").dropna()
                samples.append(vals)
                bin_ns[label] = int(vals.shape[0])
            valid_bins = [s for s in samples if len(s) > 0]
            if len(valid_bins) >= 2 and sum(len(s) for s in valid_bins) >= 4:
                H, p = stats.kruskal(*valid_bins, nan_policy="omit")
            else:
                H, p = np.nan, np.nan
            row: Dict[str, object] = {**{gc: g for gc, g in zip(group_cols, group_key if isinstance(group_key, tuple) else (group_key,))}}
            row.update({
                "metric": value_col,
                "predictor": pred,
                "H": H,
                "p_value": p,
                **{f"n_{label}": n for label, n in bin_ns.items()},
            })
            rows.append(row)
    return pd.DataFrame(rows)


def _make_bins(demo: pd.DataFrame) -> Dict[str, Tuple[pd.Series, List[str]]]:
    """Define ordinal bins for demographics; returns mapping predictor -> (labels, order)."""
    # Age bins: <=30, 31–40, >40
    age_bins = pd.cut(demo["age"], bins=[-np.inf, 30, 40, np.inf], labels=["<=30", "31-40", "41+"], include_lowest=True)
    # Experience bins (1–7 Likert): 1–3 low, 4–5 medium, 6–7 high
    def likert_bins(series: pd.Series) -> pd.Series:
        vals = pd.to_numeric(series, errors="coerce")
        labels = pd.Series(index=series.index, dtype="object")
        labels[(vals >= 1) & (vals <= 3)] = "low (1-3)"
        labels[(vals >= 4) & (vals <= 5)] = "mid (4-5)"
        labels[(vals >= 6) & (vals <= 7)] = "high (6-7)"
        return labels
    xr_bins = likert_bins(demo["xr_experience"])
    vg_bins = likert_bins(demo["videogame_experience"])
    return {
        "age": (age_bins, ["<=30", "31-40", "41+"]),
        "xr_experience": (xr_bins, ["low (1-3)", "mid (4-5)", "high (6-7)"]),
        "videogame_experience": (vg_bins, ["low (1-3)", "mid (4-5)", "high (6-7)"]),
    }


def analyze_performance_vs_demographics(
    demographics: pd.DataFrame,
    intervals: pd.DataFrame,
    durations_overall: pd.DataFrame,
    intervals_by_task: Optional[pd.DataFrame] = None,
    durations_by_task: Optional[pd.DataFrame] = None,
) -> Dict[str, pd.DataFrame]:
    """Join performance with demographics and compute correlations and group tests."""
    # Prepare joinable tables
    demo = demographics.copy()
    perf_int = intervals.copy()
    perf_dur = durations_overall.copy()
    # Inner join on participant_id, keep condition in both perf tables
    perf_int = perf_int.merge(demo, on="participant_id", how="inner")
    perf_dur = perf_dur.merge(demo, on="participant_id", how="inner")
    predictors = ["age", "xr_experience", "videogame_experience"]
    # Spearman per condition
    corr_int = _spearman_for_groups(perf_int, "mean_interval_s", predictors, ["condition"])
    corr_dur = _spearman_for_groups(perf_dur, "mean_duration_overall_s", predictors, ["condition"])
    # Binned group comparisons (Kruskal)
    bins = _make_bins(demo)
    group_int = _kruskal_by_bins(perf_int, "mean_interval_s", bins, ["condition"])
    group_dur = _kruskal_by_bins(perf_dur, "mean_duration_overall_s", bins, ["condition"])
    results = {
        "corr_intervals": corr_int,
        "corr_durations": corr_dur,
        "group_intervals": group_int,
        "group_durations": group_dur,
        "joined_intervals": perf_int,
        "joined_durations": perf_dur,
    }
    # Optional: by-task analyses
    if intervals_by_task is not None and not intervals_by_task.empty:
        perf_int_task = intervals_by_task.merge(demo, on="participant_id", how="inner")
        results["corr_intervals_by_task"] = _spearman_for_groups(perf_int_task, "mean_interval_s", predictors, ["condition", "task"])
        results["group_intervals_by_task"] = _kruskal_by_bins(perf_int_task, "mean_interval_s", bins, ["condition", "task"])
    if durations_by_task is not None and not durations_by_task.empty:
        perf_dur_task = durations_by_task.merge(demo, on="participant_id", how="inner")
        results["corr_durations_by_task"] = _spearman_for_groups(perf_dur_task, "mean_duration_s", predictors, ["condition", "task"])
        results["group_durations_by_task"] = _kruskal_by_bins(perf_dur_task, "mean_duration_s", bins, ["condition", "task"])
    return results


