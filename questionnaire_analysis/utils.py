import re
from typing import List

NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")


def normalize_text(value: str) -> str:
    if value is None:
        return ""
    lowered = value.strip().lower()
    lowered = lowered.replace("�", " ")
    lowered = lowered.replace(" ", " ")  # non-breaking space
    lowered = NON_ALNUM_RE.sub(" ", lowered)
    return re.sub(r"\s+", " ", lowered).strip()


def startswith_normalized(text: str, prefix: str) -> bool:
    return normalize_text(text).startswith(normalize_text(prefix))


def find_columns_by_prefix(columns: List[str], prefix: str) -> List[str]:
    pref = normalize_text(prefix)
    return [c for c in columns if normalize_text(c).startswith(pref)]


def split_rank_order(order_text: str) -> List[str]:
    if not isinstance(order_text, str):
        return []
    # Expect strings like "World;Head;Arm;Torso;" possibly with quotes
    parts = [p.strip().strip('"\'') for p in re.split(r"[;,]", order_text) if p.strip()]
    # Normalize capitalization and common typos
    canonical = []
    for p in parts:
        lower = p.lower()
        if lower.startswith("worl"):
            canonical.append("World")
        elif lower.startswith("hea"):
            canonical.append("Head")
        elif lower.startswith("tor") or lower.startswith("tors"):
            canonical.append("Torso")
        elif lower.startswith("arm"):
            canonical.append("Arm")
    return canonical
