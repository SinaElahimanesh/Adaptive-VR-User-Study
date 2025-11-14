from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")


def _read_questionnaire_csv_robust(path: str) -> pd.DataFrame:
    last_err: Optional[Exception] = None
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            df = pd.read_csv(path, sep=";", engine="python", encoding=enc)
            df.columns = [c.replace("\xa0", " ").strip() for c in df.columns]
            return df
        except Exception as e:
            last_err = e
            continue
    raise last_err if last_err else RuntimeError("Failed to read questionnaire CSV")


def _parse_leading_int(value: object) -> Optional[int]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    s = str(value).strip()
    num = ""
    for ch in s:
        if ch.isdigit():
            num += ch
        else:
            break
    if not num:
        return None
    try:
        return int(num)
    except ValueError:
        return None


def build_participant_demographics(questionnaire_csv: str) -> pd.DataFrame:
    df = _read_questionnaire_csv_robust(questionnaire_csv)
    # Canonical column names
    pid_col = "Participant ID"
    age_col = "Age"
    gender_col = "Gender"
    lang_col = "Language"
    xr_col = "Answer the following statements, from a scale of 1 to 7:.Experience with XR devices (VR/AR/MR)"
    vg_col = "Answer the following statements, from a scale of 1 to 7:.Experience with videogames"
    # Fallback name matching
    for want in [pid_col, age_col, gender_col, lang_col]:
        if want not in df.columns:
            for c in df.columns:
                if c.lower().startswith(want.lower().split()[0]):
                    df.rename(columns={c: want}, inplace=True)
                    break
    demo = pd.DataFrame({
        "participant_id": pd.to_numeric(df.get(pid_col), errors="coerce").astype("Int64"),
        "age": pd.to_numeric(df.get(age_col), errors="coerce"),
        "gender": df.get(gender_col),
        "language": df.get(lang_col),
        "xr_experience": df.get(xr_col).apply(_parse_leading_int) if xr_col in df.columns else None,
        "videogame_experience": df.get(vg_col).apply(_parse_leading_int) if vg_col in df.columns else None,
    })
    # Aggregate to participant level (first non-null per participant)
    agg_funcs = {
        "age": "first",
        "gender": "first",
        "language": "first",
        "xr_experience": "first",
        "videogame_experience": "first",
    }
    demo = (demo.dropna(subset=["participant_id"])
                .groupby("participant_id", as_index=False)
                .agg(agg_funcs))
    # Normalize gender strings
    if "gender" in demo.columns:
        demo["gender"] = demo["gender"].astype(str).str.strip().str.title().replace({
            "M": "Male",
            "F": "Female",
        })
    return demo


def compute_demographic_summaries(demo: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    # Summary row
    summary = pd.DataFrame([{
        "num_participants": int(demo["participant_id"].nunique()),
        "age_mean": float(demo["age"].dropna().mean()) if "age" in demo else np.nan,
        "age_std": float(demo["age"].dropna().std()) if "age" in demo else np.nan,
        "age_min": float(demo["age"].dropna().min()) if "age" in demo else np.nan,
        "age_max": float(demo["age"].dropna().max()) if "age" in demo else np.nan,
        "xr_experience_mean": float(pd.to_numeric(demo.get("xr_experience"), errors="coerce").dropna().mean()) if "xr_experience" in demo else np.nan,
        "xr_experience_std": float(pd.to_numeric(demo.get("xr_experience"), errors="coerce").dropna().std()) if "xr_experience" in demo else np.nan,
        "videogame_experience_mean": float(pd.to_numeric(demo.get("videogame_experience"), errors="coerce").dropna().mean()) if "videogame_experience" in demo else np.nan,
        "videogame_experience_std": float(pd.to_numeric(demo.get("videogame_experience"), errors="coerce").dropna().std()) if "videogame_experience" in demo else np.nan,
    }])
    # Gender counts
    gender = (demo["gender"].dropna()
              .value_counts().rename_axis("gender").reset_index(name="count"))
    # Language counts
    language = (demo["language"].dropna()
                .value_counts().rename_axis("language").reset_index(name="count"))
    # Experience distributions (1..7)
    def likert_counts(series: pd.Series, colname: str) -> pd.DataFrame:
        vals = pd.to_numeric(series, errors="coerce")
        counts = vals.value_counts(dropna=True).sort_index()
        out = counts.rename_axis(colname).reset_index(name="count")
        return out
    xr_dist = likert_counts(demo.get("xr_experience"), "xr_experience") if "xr_experience" in demo else pd.DataFrame()
    vg_dist = likert_counts(demo.get("videogame_experience"), "videogame_experience") if "videogame_experience" in demo else pd.DataFrame()
    return {
        "summary": summary,
        "gender": gender,
        "language": language,
        "xr_distribution": xr_dist,
        "videogame_distribution": vg_dist,
        "participants": demo.copy(),
    }


def _ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def plot_demographics(demo: pd.DataFrame, outdir: str) -> Dict[str, str]:
    _ensure_dir(outdir)
    fig_paths: Dict[str, str] = {}
    # Age histogram
    if "age" in demo and demo["age"].notna().any():
        plt.figure(figsize=(6, 4))
        sns.histplot(demo["age"].dropna(), bins=8, kde=False)
        plt.xlabel("Age")
        plt.ylabel("Count")
        plt.title("Age distribution")
        f = str(Path(outdir) / "age_histogram.png")
        plt.tight_layout()
        plt.savefig(f, dpi=150)
        plt.close()
        fig_paths["age_histogram"] = f
    # Gender bar
    if "gender" in demo and demo["gender"].notna().any():
        plt.figure(figsize=(5, 4))
        sns.countplot(y=demo["gender"].dropna())
        plt.xlabel("Count")
        plt.ylabel("Gender")
        plt.title("Gender distribution")
        f = str(Path(outdir) / "gender_distribution.png")
        plt.tight_layout()
        plt.savefig(f, dpi=150)
        plt.close()
        fig_paths["gender_distribution"] = f
    # XR and videogame experience bars
    for col, fname in [("xr_experience", "xr_experience_distribution.png"),
                       ("videogame_experience", "videogame_experience_distribution.png")]:
        if col in demo and pd.to_numeric(demo[col], errors="coerce").notna().any():
            plt.figure(figsize=(6, 4))
            vals = pd.to_numeric(demo[col], errors="coerce").dropna().astype(int)
            sns.countplot(x=vals, order=sorted(vals.unique()))
            plt.xlabel(col.replace("_", " ").title() + " (1–7)")
            plt.ylabel("Count")
            plt.title(col.replace("_", " ").title() + " distribution")
            f = str(Path(outdir) / fname)
            plt.tight_layout()
            plt.savefig(f, dpi=150)
            plt.close()
            fig_paths[col] = f
    return fig_paths


