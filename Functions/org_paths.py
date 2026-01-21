# org_paths.py
from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd


def build_emp_all_paths(
    df: pd.DataFrame,
    *,
    emp_col: str,
    sup_col: str,
    l1_col: str,
    sep: str = " | ",
    path_sep: str = " || ",
) -> pd.DataFrame:
    """
    Adds:
      - AllPath_IDs
      - AllPath_L1Paths
      - Is_in_layer_1_3
      - Is_in_last_layer
      - Is_Leaf

    Implementation:
      - Build all root->leaf ID paths once.
      - For each employee, collect the paths that contain that employee.
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
        if s.lower() in {"nan", "none"}:
            return None
        return s or None

    emp_ids = out[emp_col].map(_canon_id)
    sup_ids = out[sup_col].map(_canon_id)

    out["_emp_id_"] = emp_ids
    out["_sup_id_"] = sup_ids

    # adjacency manager -> direct reports
    mgr_to_reports: Dict[str, List[str]] = {}
    all_emp: Set[str] = set([e for e in emp_ids.tolist() if e])

    for e, m in zip(emp_ids.tolist(), sup_ids.tolist()):
        if not e or not m:
            continue
        if e == m:
            continue
        mgr_to_reports.setdefault(m, []).append(e)

    # leaf flag
    out["Is_Leaf"] = out["_emp_id_"].apply(lambda e: bool(e) and (e not in mgr_to_reports))

    # roots = employees whose manager is missing or not in emp set
    roots = []
    for e, m in zip(emp_ids.tolist(), sup_ids.tolist()):
        if not e:
            continue
        if (not m) or (m not in all_emp):
            roots.append(e)
    roots = list(dict.fromkeys(roots))

    # map emp_id -> l1 string
    id_to_l1: Dict[str, str] = {}
    for e, l1 in zip(emp_ids.tolist(), out[l1_col].astype(str).tolist()):
        if e and e not in id_to_l1:
            id_to_l1[e] = (l1 or "").strip()

    # build all root->leaf paths
    all_paths: List[List[str]] = []

    def dfs(node: str, stack: List[str], visiting: Set[str]) -> None:
        if node in visiting:
            return  # cycle guard
        visiting.add(node)
        stack.append(node)

        kids = mgr_to_reports.get(node, [])
        if not kids:
            all_paths.append(stack.copy())
        else:
            for ch in kids:
                dfs(ch, stack, visiting)

        stack.pop()
        visiting.remove(node)

    for r in roots:
        dfs(r, [], set())

    # index: emp -> list of paths that contain them
    emp_to_paths: Dict[str, List[List[str]]] = {e: [] for e in all_emp}
    for p in all_paths:
        s = set(p)
        for e in s:
            if e in emp_to_paths:
                emp_to_paths[e].append(p)

    # build columns
    allpath_ids_col = []
    allpath_l1_col = []
    is_layer_1_3_col = []
    is_last_layer_col = []

    for e in out["_emp_id_"].tolist():
        paths = emp_to_paths.get(e or "", [])
        if not e or not paths:
            allpath_ids_col.append("")
            allpath_l1_col.append("")
            is_layer_1_3_col.append(False)
            is_last_layer_col.append(False)
            continue

        # for each path, find position (1-indexed)
        in_1_3 = False
        in_last = False

        ids_strs = []
        l1_strs = []
        for p in paths:
            try:
                pos = p.index(e) + 1
            except ValueError:
                continue
            if pos <= 3:
                in_1_3 = True
            if pos == len(p):
                in_last = True

            ids_strs.append(sep.join(p))
            l1_strs.append(sep.join(id_to_l1.get(x, "") for x in p))

        allpath_ids_col.append(path_sep.join(ids_strs))
        allpath_l1_col.append(path_sep.join(l1_strs))
        is_layer_1_3_col.append(in_1_3)
        is_last_layer_col.append(in_last)

    out["AllPath_IDs"] = allpath_ids_col
    out["AllPath_L1Paths"] = allpath_l1_col
    out["Is_in_layer_1_3"] = pd.Series(is_layer_1_3_col, index=out.index, dtype=bool)
    out["Is_in_last_layer"] = pd.Series(is_last_layer_col, index=out.index, dtype=bool)

    out.drop(columns=["_emp_id_", "_sup_id_"], inplace=True, errors="ignore")
    return out
