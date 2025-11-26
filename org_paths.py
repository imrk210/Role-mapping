from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import re
import numpy as np
import pandas as pd

def build_supervisor_paths(
    df: pd.DataFrame,
    emp_col: Optional[str] = "employee_id",
    sup_col: Optional[str] = "supervisor_id",
    *,
    role_col: Optional[str] = "Updated_Role type (Commercial/ Non-Commercial)",
    l1_col: Optional[str]   = "Mapped_L1",
    sep: str = " | ",
    unknown_label: str = "Unknown",
    max_depth: Optional[int] = None,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, List[str]]:
    """Add supervisor_N..1 + emp + Path_IDs/Path_RoleType/Path_Mapped_L1/Has_Cycle.
       Works even if df.index has duplicates (uses .iloc everywhere)."""

    # ---------- helpers ----------
    def _norm(s: str) -> str:
        return re.sub(r"[^a-z0-9]", "", s.lower())

    def _find_col(df: pd.DataFrame, preferred: Optional[str], candidates: List[str], what: str) -> str:
        if preferred and preferred in df.columns:
            return preferred
        norm_map = {_norm(c): c for c in df.columns}
        if preferred:
            k = _norm(preferred)
            if k in norm_map:
                return norm_map[k]
        for cand in candidates:
            k = _norm(cand)
            if k in norm_map:
                return norm_map[k]
        raise KeyError(f"Missing required column for {what}: tried {preferred!r} and {candidates}")

    def _first_scalar(x):
        """Collapse Series/list/array to one scalar (first non-NA), else None."""
        if isinstance(x, pd.Series):
            x = x.dropna()
            return _first_scalar(x.iloc[0]) if not x.empty else None
        if isinstance(x, (list, tuple, set)):
            for v in x:
                s = _first_scalar(v)
                if s is not None:
                    return s
            return None
        if isinstance(x, (np.ndarray,)):
            return _first_scalar(x.flat[0]) if x.size else None
        return x

    def _canon_id(x) -> Optional[str]:
        x = _first_scalar(x)
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return None
        # normalize ints like 1010473.0 -> "1010473"
        try:
            xf = float(x)
            if np.isfinite(xf) and xf.is_integer():
                return str(int(xf))
        except Exception:
            pass
        s = str(x).strip()
        return s or None

    # ---------- columns ----------
    emp_col = _find_col(
        df, emp_col,
        ["employee_id","employee id","emp_id","emp id","empid","eid",
         "worker_id","worker id","person_id","person id","personnumber"],
        "employee id"
    )
    sup_col = _find_col(
        df, sup_col,
        ["supervisor_id","supervisor id","manager_id","manager id","mgr_id","mgr id",
         "reports_to","reports to","reports_to_id","line_manager_id","line manager id",
         "supervisor","manager"],
        "supervisor id"
    )

    if verbose:
        print(f"[org_paths] using columns → employee: {emp_col!r} | supervisor: {sup_col!r}")

    out = df.copy()

    # canonical id columns (positional)
    out["_emp_id_"] = out[emp_col].map(_canon_id)
    out["_sup_id_"] = out[sup_col].map(_canon_id)

    # dicts for quick lookups (keys are strings)
    id_to_sup: Dict[Optional[str], Optional[str]] = dict(zip(out["_emp_id_"], out["_sup_id_"]))

    id_to_role: Dict[str, str] = {}
    if role_col and role_col in out.columns:
        emp_vals = out["_emp_id_"].values
        role_vals = out[role_col].values
        for e, r in zip(emp_vals, role_vals):
            if e is not None and e not in id_to_role:
                rr = _first_scalar(r)
                id_to_role[e] = unknown_label if rr is None or pd.isna(rr) else str(rr)

    id_to_l1: Dict[str, str] = {}
    if l1_col and l1_col in out.columns:
        emp_vals = out["_emp_id_"].values
        l1_vals = out[l1_col].values
        for e, l in zip(emp_vals, l1_vals):
            if e is not None and e not in id_to_l1:
                ll = _first_scalar(l)
                id_to_l1[e] = unknown_label if ll is None or pd.isna(ll) else str(ll)

    def ascend_chain(emp_id: Optional[str]) -> Tuple[List[str], bool]:
        """([sup1, sup2, ...], has_cycle). sup1 = direct manager."""
        if emp_id is None:
            return [], False
        seen = {emp_id}
        cur = id_to_sup.get(emp_id)
        chain: List[str] = []
        while cur is not None and cur not in seen:
            chain.append(cur)
            seen.add(cur)
            cur = id_to_sup.get(cur)
        cycle = cur is not None and cur in seen
        return chain, cycle

    # compute chains using positions, never labels
    n = len(out)
    chains: List[List[str]] = [[] for _ in range(n)]
    cycles: List[bool]      = [False for _ in range(n)]
    max_len = 0
    emp_vals = out["_emp_id_"].to_numpy()
    for pos in range(n):
        sups, cyc = ascend_chain(emp_vals[pos])
        chains[pos] = sups
        cycles[pos] = cyc
        if len(sups) > max_len:
            max_len = len(sups)

    depth = max_depth if (isinstance(max_depth, int) and max_depth >= 0) else max_len
    sup_cols: List[str] = [f"supervisor_{k}" for k in range(depth, 0, -1)]
    for c in sup_cols:
        out[c] = None
    out["emp"] = out["_emp_id_"]

    # column locations for fast positional set
    col_loc = {c: out.columns.get_loc(c) for c in sup_cols + ["emp"]}

    def full_path_ids(pos: int) -> List[Optional[str]]:
        sups = chains[pos]                 # [sup1, sup2, ...]
        far_to_near = list(reversed(sups)) # [supN ... sup1]
        return far_to_near + [emp_vals[pos]]

    # fill supervisor_N..1 with positions
    for pos in range(n):
        ids = full_path_ids(pos)
        far_to_near = ids[:-1]
        if len(far_to_near) < depth:
            far_to_near = [None] * (depth - len(far_to_near)) + far_to_near
        for col, val in zip(sup_cols, far_to_near[-depth:]):
            out.iat[pos, col_loc[col]] = val
        # emp already filled from _emp_id_

    def safe_label(id_: Optional[str], mapping: Dict[str, str]) -> str:
        id_ = _canon_id(id_)
        if id_ is None:
            return unknown_label
        return mapping.get(id_, unknown_label)

    # build path strings (IDs / RoleType / L1)
    path_ids, path_role, path_l1 = [], [], []
    for pos in range(n):
        ids = full_path_ids(pos)  # [supN ... sup1, emp]
        path_ids.append(sep.join("" if x is None else str(x) for x in ids))
        path_role.append(sep.join(safe_label(x, id_to_role) for x in ids) if id_to_role else "")
        path_l1.append(sep.join(safe_label(x, id_to_l1) for x in ids)   if id_to_l1   else "")

    out["Path_IDs"] = path_ids
    if role_col and role_col in out.columns:
        out["Path_RoleType"] = path_role
    if l1_col and l1_col in out.columns:
        out["Path_Mapped_L1"] = path_l1
    out["Has_Cycle"] = pd.Series(cycles, index=out.index, dtype=bool)

    out.drop(columns=["_emp_id_", "_sup_id_"], inplace=True, errors="ignore")
    return out, sup_cols
