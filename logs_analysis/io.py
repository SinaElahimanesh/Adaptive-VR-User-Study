from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


def discover_participants(logs_root: str) -> List[str]:
    root = Path(logs_root)
    return sorted([p.name for p in root.iterdir() if p.is_dir() and p.name.startswith('Participant_')])


def _read_csv_flexible(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Normalize columns: remove spaces
    df.columns = [c.strip().replace(' ', '') for c in df.columns]
    # Standardize expected columns
    rename = {
        'UIName': 'UIName',
        'EventType': 'EventType',
        'CoordinateSystem': 'CoordinateSystem',
        'Correct': 'Correct',
        'Timestamp': 'Timestamp',
        'Duration': 'Duration',
    }
    for need in ['UIName', 'EventType', 'CoordinateSystem', 'Timestamp']:
        if need not in df.columns:
            # attempt case-insensitive match
            for c in df.columns:
                if c.lower() == need.lower():
                    df.rename(columns={c: need}, inplace=True)
                    break
    # Fill missing duration with 0 for Position events
    if 'Duration' in df.columns:
        df['Duration'] = pd.to_numeric(df['Duration'], errors='coerce')
    if 'EventType' in df.columns and 'Duration' in df.columns:
        df.loc[df['EventType'].astype(str).str.contains('Position', na=False) & df['Duration'].isna(), 'Duration'] = 0.0
    # Coerce numeric
    if 'Timestamp' in df.columns:
        df['Timestamp'] = pd.to_numeric(df['Timestamp'], errors='coerce')
    if 'Correct' in df.columns:
        df['Correct'] = df['Correct'].astype(str).str.strip().map({'True': True, 'False': False})
    return df


def load_ui_log(participant_dir: str, condition: str) -> pd.DataFrame:
    path = Path(participant_dir) / f"StudyUILog_{participant_dir.split('_')[-1]}_{condition}.csv"
    if not path.exists():
        candidates = list(Path(participant_dir).glob(f"StudyUILog_*_{condition}.csv"))
        if not candidates:
            return pd.DataFrame()
        path = candidates[0]
    df = _read_csv_flexible(path)
    df['condition'] = condition
    df['participant_dir'] = Path(participant_dir).name
    return df


def load_positioning_log(participant_dir: str, condition: str) -> pd.DataFrame:
    path = Path(participant_dir) / f"StudyUIPositioningLog_{participant_dir.split('_')[-1]}_{condition}.csv"
    if not path.exists():
        candidates = list(Path(participant_dir).glob(f"StudyUIPositioningLog_*_{condition}.csv"))
        if not candidates:
            return pd.DataFrame()
        path = candidates[0]
    df = _read_csv_flexible(path)
    df['condition'] = condition
    df['participant_dir'] = Path(participant_dir).name
    return df


def scan_motion_log(participant_dir: str, condition: str, max_frames: int = 10000) -> Dict[str, float]:
    candidates = list(Path(participant_dir).glob(f"StudyMotionLog_*_{condition}.json"))
    if not candidates:
        return {}
    path = candidates[0]
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    rate = float(data.get('captureRateHz', 0))
    frames = data.get('frames', [])
    total_frames = len(frames)
    sample = frames[:min(total_frames, max_frames)]
    if not sample:
        return {"capture_rate_hz": rate, "total_frames": total_frames, "duration_seconds": 0.0, "fraction_valid": 0.0}
    ts0 = sample[0].get('timestamp', 0.0)
    tsN = sample[-1].get('timestamp', 0.0)
    valid = sum(1 for fr in sample if fr.get('isValid', False))
    frac_valid = valid / len(sample)
    approx_duration = (tsN - ts0) + max(0.0, (total_frames - len(sample)) / rate if rate else 0.0)
    return {
        "capture_rate_hz": rate,
        "total_frames": float(total_frames),
        "duration_seconds": float(approx_duration),
        "fraction_valid": float(frac_valid),
    }
