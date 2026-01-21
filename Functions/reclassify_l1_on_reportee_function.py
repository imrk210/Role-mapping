# reclassify_l1_on_reportee_function.py
from __future__ import annotations

from typing import Dict, List, Optional, Set
import numpy as np
import pandas as pd

CANONICAL_7 = [
    "Sales",
    "Marketing",
    "Customer Success",
    "Customer Support",
    "Pricing",
    "Revenue Operations (RevOps)",
    "Product Management",
]
CANON_SET = set(CANONICAL_7)


def reclassify_l1_using_descendants_and_supervisors(
    df: pd.DataFrame,
    *,
    emp_col: str,
    sup_col: str,
    l1_col: str,
    l1_conf_col: str,
    allpath_ids_col: str,
    is_layer_1_3_col: str,
    is_last_layer_col: str,
    threshold: float = 0.8,
    out_reclass_col: str = "Reclassified_L1",
    out_final_col: str = "Final_Mapped_L1",
    min_votes: int = 2,
    max_sup_layers: int = 4,
    sep: str = " | ",
    path_sep: str = " || ",
) -> pd.DataFrame:
    """
    Applies reportee-based reclassification to BOTH commercial and non-commercial rows
    when L1 confidence < threshold and NOT in layer 1-3.

    Voting is always ONLY across CANONICAL_7 buckets.
    If vote decides, overwrite Final_Mapped_L1 to that canonical bucket.

    Supervisor fallback applies only if the employee is in bottom-most layer.
    """
    out = df.copy()

    def _canon_id(x) -> Optional[str]:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return None
        try:
            xf = float(x)
            if np.isfinite(xf) and xf.is_integer():
                return str(int(xf))
        except Exception:
            pass
        s = str(x).strip()
        if not s or s.lower() in {"nan", "none"}:
            return None
        return s

    emp_ids = out[emp_col].map(_canon_id)
    sup_ids = out[sup_col].map(_canon_id)

    # adjacency manager -> direct reports
    mgr_to_reports: Dict[str, List[str]] = {}
    for e, m in zip(emp_ids.tolist(), sup_ids.tolist()):
        if not e or not m or e == m:
            continue
        mgr_to_reports.setdefault(m, []).append(e)

    # L1 lookup (prefer out_final_col if it already exists; else l1_col)
    base_l1_source_col = out_final_col if out_final_col in out.columns else l1_col

    id_to_l1: Dict[str, str] = {}
    for e, l1 in zip(emp_ids.tolist(), out[base_l1_source_col].astype(str).tolist()):
        if e and e not in id_to_l1:
            id_to_l1[e] = (l1 or "").strip()

    def descendants(eid: str) -> List[str]:
        """All descendants BFS/DFS."""
        seen: Set[str] = set()
        stack = [eid]
        ds: List[str] = []
        while stack:
            node = stack.pop()
            for ch in mgr_to_reports.get(node, []):
                if ch in seen:
                    continue
                seen.add(ch)
                ds.append(ch)
                stack.append(ch)
        return ds

    def parse_paths(cell: object) -> List[List[str]]:
        """Parse AllPath_IDs cell into list of paths (list[str])."""
        if cell is None:
            return []
        s = str(cell)
        if not s or s.strip().lower() in {"nan", "none"}:
            return []
        raw_paths = s.split(path_sep)
        paths: List[List[str]] = []
        for rp in raw_paths:
            p = [x.strip() for x in rp.split(sep) if x.strip()]
            if p:
                # canonicalize IDs inside paths too
                p2 = [_canon_id(x) for x in p]
                p2 = [x for x in p2 if x]
                if p2:
                    paths.append(p2)
        return paths

    # initialize outputs
    out[out_reclass_col] = ""
    out[out_final_col] = out[l1_col]

    conf = pd.to_numeric(out[l1_conf_col], errors="coerce").fillna(1.0)
    low_conf = conf < float(threshold)

    # ✅ APPLY TO BOTH commercial + non-commercial:
    mask = low_conf & (~out[is_layer_1_3_col].fillna(False).astype(bool))

    for i in out.index[mask]:
        eid = emp_ids.loc[i]
        if not eid:
            continue

        chosen: Optional[str] = None

        # 1) descendant vote (ONLY canonical 7)
        votes: List[str] = []
        for d in descendants(eid):
            l1 = id_to_l1.get(d, "")
            if l1 in CANON_SET:
                votes.append(l1)

        if len(votes) >= min_votes:
            vc = pd.Series(votes).value_counts()
            top = str(vc.index[0])
            top_n = int(vc.iloc[0])
            second_n = int(vc.iloc[1]) if len(vc) > 1 else 0
            if top in CANON_SET and top_n >= min_votes and (top_n - second_n) >= 1:
                chosen = top

        # 2) supervisor fallback only if bottom-most layer (ONLY canonical 7)
        if chosen is None and bool(out.loc[i, is_last_layer_col]):
            paths = parse_paths(out.loc[i, allpath_ids_col])
            sup_votes: List[str] = []

            for p in paths:
                if eid not in p:
                    continue
                pos = p.index(eid)
                start = max(0, pos - max_sup_layers)
                sups = p[start:pos]
                for sid in sups:
                    l1s = id_to_l1.get(sid, "")
                    if l1s in CANON_SET:
                        sup_votes.append(l1s)

            if len(sup_votes) >= min_votes:
                vc = pd.Series(sup_votes).value_counts()
                top = str(vc.index[0])
                top_n = int(vc.iloc[0])
                second_n = int(vc.iloc[1]) if len(vc) > 1 else 0
                if top in CANON_SET and top_n >= min_votes and (top_n - second_n) >= 1:
                    chosen = top

        if chosen:
            out.at[i, out_reclass_col] = chosen
            out.at[i, out_final_col] = chosen

    return out
