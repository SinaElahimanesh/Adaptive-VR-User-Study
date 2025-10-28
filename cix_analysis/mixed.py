from __future__ import annotations

from typing import Dict
import numpy as np
import pandas as pd

from .config import TASKS


def triangulate_findings(
    desc_by_condition: pd.DataFrame,
    justification_mentions: pd.DataFrame,
    theme_counts: pd.DataFrame,
    comments: pd.DataFrame,
) -> Dict[str, pd.DataFrame]:
    # Top anchors by numeric mean per condition x task
    top_numeric = (
        desc_by_condition
        .sort_values(["condition", "task", "mean"], ascending=[True, True, False])
        .groupby(["condition", "task"]).first().reset_index()[["condition", "task", "anchor", "mean"]]
        .rename(columns={"anchor": "top_anchor_numeric", "mean": "top_mean"})
    )

    # Top anchors by mentions in justifications per condition x task
    jm = justification_mentions.copy()
    jm_long = []
    for _, r in jm.iterrows():
        for anchor, col in {
            "World": "mentions_world",
            "Head": "mentions_head",
            "Torso": "mentions_torso",
            "Arm": "mentions_arm",
        }.items():
            jm_long.append({
                "condition": r.get("condition"),
                "task": r.get("task"),
                "anchor": anchor,
                "hits": r.get(col, 0),
            })
    jm_long = pd.DataFrame(jm_long)
    if not jm_long.empty:
        top_mentions = (
            jm_long.groupby(["condition", "task", "anchor"])['hits'].sum().reset_index()
            .sort_values(["condition", "task", "hits"], ascending=[True, True, False])
            .groupby(["condition", "task"]).first().reset_index()[["condition", "task", "anchor", "hits"]]
            .rename(columns={"anchor": "top_anchor_mentions", "hits": "top_hits"})
        )
    else:
        top_mentions = pd.DataFrame(columns=["condition", "task", "top_anchor_mentions", "top_hits"]) 

    alignment = top_numeric.merge(top_mentions, on=["condition", "task"], how="left")
    alignment["agree"] = alignment["top_anchor_numeric"] == alignment["top_anchor_mentions"]

    # Theme prevalence by condition
    themes = theme_counts.merge(comments[["row_index", "condition"]], on="row_index", how="left")
    theme_cols = [c for c in themes.columns if c not in ["row_index", "condition"]]
    theme_prev = themes.groupby("condition")[theme_cols].sum().reset_index()
    total_rows = themes.groupby("condition")["row_index"].count().rename("n").reset_index()
    theme_prev = theme_prev.merge(total_rows, on="condition", how="left")
    for c in theme_cols:
        theme_prev[f"{c}_per_participant"] = theme_prev[c] / theme_prev["n"].replace(0, np.nan)

    return {"anchor_alignment": alignment, "theme_prevalence": theme_prev}
