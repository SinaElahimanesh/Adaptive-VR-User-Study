from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import pandas as pd

# Ensure local package import when running directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from questionnaire_analysis.demographics import (
    build_participant_demographics,
    compute_demographic_summaries,
    plot_demographics,
)


def save_table(df: pd.DataFrame, path: str) -> None:
    os.makedirs(Path(path).parent, exist_ok=True)
    df.to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute Questionnaire demographics")
    parser.add_argument("--csv", default="Questionnaire.csv", help="Path to Questionnaire.csv")
    parser.add_argument("--outdir", default="outputs/questionnaire", help="Directory to write outputs")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    demo = build_participant_demographics(args.csv)
    tables = compute_demographic_summaries(demo)
    # Save tables
    save_table(tables["summary"], os.path.join(args.outdir, "demographics_summary.csv"))
    save_table(tables["gender"], os.path.join(args.outdir, "demographics_gender.csv"))
    save_table(tables["language"], os.path.join(args.outdir, "demographics_language.csv"))
    if not tables["xr_distribution"].empty:
        save_table(tables["xr_distribution"], os.path.join(args.outdir, "demographics_xr_distribution.csv"))
    if not tables["videogame_distribution"].empty:
        save_table(tables["videogame_distribution"], os.path.join(args.outdir, "demographics_videogame_distribution.csv"))
    # Figures
    figs_dir = os.path.join(args.outdir, "figs")
    plot_demographics(demo, figs_dir)
    # Also save per-participant resolved demographics
    save_table(tables["participants"], os.path.join(args.outdir, "demographics_participants.csv"))
    print(f"Saved demographics to {args.outdir}")


if __name__ == "__main__":
    main()


