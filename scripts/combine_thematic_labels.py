#!/usr/bin/env python3
"""
Combine thematic labels from two coders (Sina and Joao) by Participant ID and Trial condition.

Input CSVs are expected in:
  - thematic_analysis/Themes-Sina.csv
  - thematic_analysis/Themes-Joao.csv

Join keys:
  - 'Participant ID'
  - 'Trial condition (To be filled by the experimenter)'

Output columns (in order):
  - 'Participant ID'
  - 'Trial condition (To be filled by the experimenter)'
  - 'Explain your decision in a few words:'
  - 'Label - Key (Sina)'
  - 'Label - Key (Joao)'
  - 'Explain your decision in a few words:1'
  - 'Label - Visual (Sina)'
  - 'Label - Visual (Joao)'
  - 'Explain your decision in a few words:2'
  - 'Label - Controls (Sina)'
  - 'Label - Controls (Joao)'

Notes:
- Explanation columns (the three "Explain your decision..." fields) are coalesced:
  we take Sina's value when present, otherwise Joao's value.
- If some input columns are missing or differently punctuated (e.g., with/without
  trailing colon), the script attempts robust matching.
"""
from __future__ import annotations

import argparse
import os
import re
from typing import Dict, Iterable, List, Tuple, Set

import pandas as pd


WORKSPACE_ROOT = "/Users/sinaelahimanesh/Documents/Saarland/CIX/Data Analysis"
DEFAULT_SINA = os.path.join(WORKSPACE_ROOT, "thematic_analysis", "Themes-Sina.csv")
DEFAULT_JOAO = os.path.join(WORKSPACE_ROOT, "thematic_analysis", "Themes-Joao.csv")
DEFAULT_OUT = os.path.join(WORKSPACE_ROOT, "outputs", "thematic_analysis", "combined_thematic_labels.csv")

KEY_COLS = [
    "Participant ID",
    "Trial condition (To be filled by the experimenter)",
]

# Base names used for matching; script tolerates variants like trailing colon, spacing, etc.
LABEL_BASE_NAMES = [
    "Label - Key",
    "Label - Visual",
    "Label - Controls",
]

EXPLAIN_BASE_NAMES = [
    "Explain your decision in a few words",    # may appear as ":" or without
    "Explain your decision in a few words:1",
    "Explain your decision in a few words:2",
]


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    return df


def candidates_for(name: str) -> List[str]:
    """
    Generate reasonable candidate variants for a column name to improve robustness
    against small punctuation differences across coders/files.
    """
    variants = set()
    base = name.strip()
    variants.add(base)
    # handle trailing colon presence/absence for the first explanation field
    if base.endswith(":1") or base.endswith(":2"):
        # exact as given; also consider without colon before number
        if ":1" in base:
            variants.add(base.replace(":1", " 1"))
        if ":2" in base:
            variants.add(base.replace(":2", " 2"))
    else:
        # try with and without trailing colon
        if not base.endswith(":"):
            variants.add(f"{base}:")
        else:
            variants.add(base[:-1])
    # minor whitespace variants around hyphens
    variants.add(base.replace(" - ", "-"))
    variants.add(base.replace("-", " - "))
    # duplicates automatically removed by set
    return list(variants)


def find_first_present(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    present = {c for c in df.columns}
    for cand in candidates:
        if cand in present:
            return cand
    return None


def ensure_key_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure the dataframe contains the canonical KEY_COLS by finding close variants
    and renaming them to the canonical names.
    """
    df = df.copy()
    rename_map: Dict[str, str] = {}
    for key in KEY_COLS:
        found = find_first_present(df, candidates_for(key))
        if found is None:
            raise KeyError(f"Missing expected key column: {key}")
        if found != key:
            rename_map[found] = key
    if rename_map:
        df = df.rename(columns=rename_map)
    return df


def load_csv_robust(path: str) -> pd.DataFrame:
    """
    Load CSV trying header row at 0, and if key columns are missing, try header at 1.
    """
    # Try header=0
    df0 = normalize_columns(pd.read_csv(path, dtype=str, keep_default_na=True, na_values=["", "NA", "NaN"], header=0))
    try:
        return ensure_key_columns(df0)
    except KeyError:
        # Try header=1 (second row as header) for files with a grouping header row
        df1 = normalize_columns(pd.read_csv(path, dtype=str, keep_default_na=True, na_values=["", "NA", "NaN"], header=1))
        return ensure_key_columns(df1)


def select_label_columns(df: pd.DataFrame, coder_suffix: str) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Keep key columns and the three label columns for a given coder.
    Returns (slim_df, mapping) where mapping is from base label name -> renamed label with suffix.
    """
    df = ensure_key_columns(df)

    keep_cols = list(KEY_COLS)
    rename_map: Dict[str, str] = {}
    for base in LABEL_BASE_NAMES:
        src = find_first_present(df, candidates_for(base))
        if src is None:
            # If a label is entirely missing, keep going; we'll create an empty column later
            continue
        renamed = f"{base} ({coder_suffix})"
        keep_cols.append(src)
        rename_map[src] = renamed

    out = df[keep_cols].rename(columns=rename_map)
    # Ensure all expected label columns exist even if missing in input (as empty)
    for base in LABEL_BASE_NAMES:
        col = f"{base} ({coder_suffix})"
        if col not in out.columns:
            out[col] = pd.NA
    return out, {b: f"{b} ({coder_suffix})" for b in LABEL_BASE_NAMES}


def select_explanation_columns(df: pd.DataFrame, coder_tag: str) -> pd.DataFrame:
    """
    Return a dataframe with key columns and explanation columns, renamed to include coder tag,
    e.g., 'Explain your decision in a few words:' -> 'Explain your decision in a few words:_sina'
    """
    df = ensure_key_columns(df)

    keep_cols = list(KEY_COLS)
    rename_map: Dict[str, str] = {}

    def target_name(base: str) -> str:
        # Use canonical targets (with colon where applicable for the first), plus suffix tag
        canonical = base
        if base == "Explain your decision in a few words":
            canonical = "Explain your decision in a few words:"
        return f"{canonical}_{coder_tag}"

    # Find sources for key, visual, controls explanations with coder-specific heuristics
    base_src = find_first_present(df, candidates_for("Explain your decision in a few words"))
    num1_src = find_first_present(df, candidates_for("Explain your decision in a few words:1"))
    num2_src = find_first_present(df, candidates_for("Explain your decision in a few words:2"))
    num3_src = find_first_present(df, candidates_for("Explain your decision in a few words:3"))

    if base_src is not None:
        keep_cols.append(base_src)
        rename_map[base_src] = target_name("Explain your decision in a few words")

    # Heuristic mapping:
    # - If :3 exists, assume Visual=:2 and Controls=:3 (Joao style)
    # - Else if :1 and :2 exist, assume Visual=:1 and Controls=:2 (Sina style)
    # - Else map whichever exists to Visual first, Controls second if available
    visual_src = None
    controls_src = None
    if num3_src is not None and num2_src is not None:
        visual_src, controls_src = num2_src, num3_src
    elif num1_src is not None and num2_src is not None:
        visual_src, controls_src = num1_src, num2_src
    elif num1_src is not None and num2_src is None and num3_src is None:
        visual_src = num1_src
    elif num2_src is not None and num1_src is None and num3_src is None:
        visual_src = num2_src
    elif num3_src is not None and num1_src is None and num2_src is None:
        controls_src = num3_src

    if visual_src is not None:
        keep_cols.append(visual_src)
        rename_map[visual_src] = target_name("Explain your decision in a few words:1")
    if controls_src is not None:
        keep_cols.append(controls_src)
        rename_map[controls_src] = target_name("Explain your decision in a few words:2")

    out = df[keep_cols].rename(columns=rename_map)
    # Ensure all explanation columns exist
    for base in EXPLAIN_BASE_NAMES:
        tgt = target_name(base)
        if tgt not in out.columns:
            out[tgt] = pd.NA
    return out


def coalesce(series_a: pd.Series, series_b: pd.Series) -> pd.Series:
    """Prefer non-null from A; otherwise B."""
    return series_a.where(series_a.notna() & (series_a.astype(str).str.strip() != ""), series_b)


def parse_label_numbers(cell: object) -> Set[int]:
    """
    Extract numeric codes from a label cell like '2, 3' into a set of ints.
    Returns empty set for NA/empty values.
    """
    if cell is None or (isinstance(cell, float) and pd.isna(cell)):
        return set()
    text = str(cell)
    numbers = re.findall(r"\d+", text)
    return {int(n) for n in numbers}


def format_number_set(values: Set[int]) -> object:
    """Format a set of ints as '1, 3, 5' or return pd.NA if empty."""
    if not values:
        return pd.NA
    return ", ".join(str(n) for n in sorted(values))


def build_output(
    sina_labels: pd.DataFrame,
    joao_labels: pd.DataFrame,
    sina_expl: pd.DataFrame,
    joao_expl: pd.DataFrame,
    how: str,
) -> pd.DataFrame:
    # Merge labels for both coders
    labels_merged = pd.merge(
        sina_labels,
        joao_labels,
        on=KEY_COLS,
        how=how,
        validate="one_to_one",
    )
    # Merge explanations separately to keep columns manageable, then join
    expl_merged = pd.merge(
        sina_expl,
        joao_expl,
        on=KEY_COLS,
        how=how,
        validate="one_to_one",
    )
    merged = pd.merge(
        labels_merged,
        expl_merged,
        on=KEY_COLS,
        how=how,
        validate="one_to_one",
    )

    # Coalesce explanations: prefer Sina, fallback to Joao
    merged["Explain your decision in a few words:"] = coalesce(
        merged["Explain your decision in a few words:_sina"],
        merged["Explain your decision in a few words:_joao"],
    )
    merged["Explain your decision in a few words:1"] = coalesce(
        merged["Explain your decision in a few words:1_sina"],
        merged["Explain your decision in a few words:1_joao"],
    )
    merged["Explain your decision in a few words:2"] = coalesce(
        merged["Explain your decision in a few words:2_sina"],
        merged["Explain your decision in a few words:2_joao"],
    )

    # Compute intersections for labels
    def compute_intersection(col_sina: str, col_joao: str, col_out: str) -> None:
        a = merged[col_sina].apply(parse_label_numbers)
        b = merged[col_joao].apply(parse_label_numbers)
        merged[col_out] = [format_number_set(x & y) for x, y in zip(a, b)]

    compute_intersection("Label - Key (Sina)", "Label - Key (Joao)", "Label - Key (Intersection)")
    compute_intersection("Label - Visual (Sina)", "Label - Visual (Joao)", "Label - Visual (Intersection)")
    compute_intersection("Label - Controls (Sina)", "Label - Controls (Joao)", "Label - Controls (Intersection)")

    # Final column order
    final_cols = [
        KEY_COLS[0],
        KEY_COLS[1],
        "Explain your decision in a few words:",
        "Label - Key (Sina)",
        "Label - Key (Joao)",
        "Label - Key (Intersection)",
        "Explain your decision in a few words:1",
        "Label - Visual (Sina)",
        "Label - Visual (Joao)",
        "Label - Visual (Intersection)",
        "Explain your decision in a few words:2",
        "Label - Controls (Sina)",
        "Label - Controls (Joao)",
        "Label - Controls (Intersection)",
    ]

    # Ensure all referenced columns exist; if a label was completely missing in both, it already exists as NA
    for col in final_cols:
        if col not in merged.columns:
            merged[col] = pd.NA

    return merged[final_cols]


def main() -> None:
    parser = argparse.ArgumentParser(description="Combine thematic labels from two coders (Sina, Joao).")
    parser.add_argument("--sina", default=DEFAULT_SINA, help="Path to Themes-Sina.csv")
    parser.add_argument("--joao", default=DEFAULT_JOAO, help="Path to Themes-Joao.csv")
    parser.add_argument("--out", default=DEFAULT_OUT, help="Path to write combined CSV")
    parser.add_argument(
        "--how",
        default="inner",
        choices=["inner", "outer", "left", "right"],
        help="Type of join on keys (default: inner)",
    )
    args = parser.parse_args()

    # Load inputs
    sina_df = load_csv_robust(args.sina)
    joao_df = load_csv_robust(args.joao)

    # Prepare label and explanation subsets
    sina_labels, _ = select_label_columns(sina_df, coder_suffix="Sina")
    joao_labels, _ = select_label_columns(joao_df, coder_suffix="Joao")
    sina_expl = select_explanation_columns(sina_df, coder_tag="sina")
    joao_expl = select_explanation_columns(joao_df, coder_tag="joao")

    combined = build_output(sina_labels, joao_labels, sina_expl, joao_expl, how=args.how)

    # Write output
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    combined.to_csv(args.out, index=False)

    print(f"Wrote {len(combined)} rows to: {args.out}")
    print(f"Join type: {args.how}")


if __name__ == "__main__":
    main()


