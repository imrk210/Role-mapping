import re
import pandas as pd
import numpy as np

def mark_role_type_match(
    df: pd.DataFrame,
    *,
    path_col: str = "Path_RoleType",
    target_col: str = "Updated_Role type (Commercial/ Non-Commercial)",
    n: int = 4,                                # 1-based starting token
    sep_pattern: str = r"\s*\|\s*",            # flexible " | " splitter
    out_col: str = "Hierarchy_Shift?",
    debug_candidate_col: str | None = None     # e.g. "Compared_Token"
) -> pd.DataFrame:
    """
    Start at the n-th token in `path_col` (1-based). If that token isn’t
    'Commercial' or 'Non-commercial', move to the next. If no suitable token
    is found, or if the path has < n tokens, compare against the LAST token.
    Set `out_col` to True if equal to `target_col` (case-insensitive), else False.
    """

    splitter = re.compile(sep_pattern)

    def _canon_role(s: str | None) -> str | None:
        """Normalize to 'commercial' or 'non-commercial' when possible; else raw lower text."""
        if s is None:
            return None
        t = str(s)
        # strip and normalize whitespace/dashes
        t = t.strip().lower()
        t = re.sub(r"[–—-]", "-", t)            # unify dash variants
        t = re.sub(r"\s+", " ", t)              # collapse spaces
        t = t.replace("non commercial", "non-commercial").replace("noncommercial", "non-commercial")
        if t in {"commercial", "non-commercial"}:
            return t
        return t  # return raw-lower text for fallback comparisons

    # Tokenize each path into a clean list
    tokens_series = (
        df[path_col]
          .astype(str)
          .fillna("")
          .apply(lambda s: [tok.strip() for tok in splitter.split(s) if tok is not None and tok.strip() != ""])
    )

    # Normalize target (but keep as string even if it isn't exactly commercial/non-commercial)
    target_norm = df[target_col].apply(_canon_role)

    def pick_candidate(tokens: list[str]) -> str | None:
        if not tokens:
            return None
        # start index: if len < n, start at last (n-1 clamped to last index)
        start_ix = min(max(n - 1, 0), len(tokens) - 1)

        # scan forward for first recognizable role; otherwise fall back to the last token
        for i in range(start_ix, len(tokens)):
            cand = _canon_role(tokens[i])
            if cand in {"commercial", "non-commercial"}:
                return cand

        # fallback: last token (normalized/raw-lower)
        return _canon_role(tokens[-1])

    candidate = tokens_series.apply(pick_candidate)

    # Compare normalized strings directly
    df[out_col] = (candidate == target_norm)

    if debug_candidate_col:
        df[debug_candidate_col] = candidate

    return df
