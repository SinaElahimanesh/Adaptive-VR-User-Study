from __future__ import annotations

import pandas as pd
from typing import Tuple
from .utils import normalize_text


def load_questionnaire(csv_path: str) -> pd.DataFrame:
    # Try common encodings; fallback to replacing undecodable bytes
    tried = []
    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin1"):
        try:
            df = pd.read_csv(csv_path, sep=";", encoding=enc, dtype=str, engine="python")
            break
        except UnicodeDecodeError as e:
            tried.append((enc, str(e)))
            df = None
    if df is None:
        # Last resort: replace undecodable bytes
        df = pd.read_csv(
            csv_path,
            sep=";",
            encoding="utf-8",
            encoding_errors="replace",
            dtype=str,
            engine="python",
        )

    # Trim whitespace and sanitize non-breaking spaces in headers and cells
    df.columns = [c.strip().replace("\xa0", " ") for c in df.columns]
    df = df.applymap(lambda x: x.replace("\xa0", " ").strip() if isinstance(x, str) else x)

    # Preserve original columns; also add a normalized map for convenience
    df.attrs["normalized_columns"] = {c: normalize_text(c) for c in df.columns}
    return df


def save_table(df: pd.DataFrame, path: str) -> None:
    df.to_csv(path, index=False)
