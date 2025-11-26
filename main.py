#!/usr/bin/env python3
from __future__ import annotations
from typing import Optional, Dict
import os
import pandas as pd
import numpy as np

from dotenv import load_dotenv
load_dotenv()

from org_paths import build_supervisor_paths
from Hierarchy_reclassification import mark_role_type_match

from batch_mapping import (
    batch_map_employees,
    load_taxonomy_from_frames,
    extract_descriptions_from_taxonomy_df,
    keywords_by_l1_from_examples,
)

# ---------------- tiny helpers ----------------
def _to_bool(v: Optional[str], default: bool = False) -> bool:
    if v is None:
        return default
    return str(v).strip().lower() in {"1", "true", "t", "yes", "y"}

def _c(c: str, s: str) -> str:
    return f"\033[{c}m{s}\033[0m"

INFO = "36"   # cyan
OK   = "32"   # green
WARN = "33"
ERR  = "31"

# ---------------- main runner ----------------
def run_mapping_with_files(
    census_path: str,
    taxonomy_xlsx: str,
    *,
    title_col: str,
    bu_col: Optional[str] = None,
    use_ai_fallback: bool = True,     # kept for compatibility; not used in batch mode
    model: str = "gpt-4o",
    temperature: float = 0.0,
    verbose: bool = True,
    min_confidence: float = 0.85,
    examples_col: str = "Final Titles",   # <- align with taxonomy
) -> pd.DataFrame:

    # --- read census ---
    print(_c(INFO, f"[load] census: {census_path}"))
    if census_path.lower().endswith((".xlsx", ".xls")):
        employees_df = pd.read_excel(census_path)
    else:
        employees_df = pd.read_csv(census_path)

    # de-duplicate on role-defining fields (keep IDs for later merge)
    unique_roles = (
        employees_df
        .drop(columns=["employee_id", "supervisor_id"], errors="ignore")
        .drop_duplicates()
        .sort_values(by=title_col)
    )
    print(_c(OK, f"[load] census rows (unique roles): {len(unique_roles)}"))

    # --- read taxonomy workbook ---
    print(_c(INFO, f"[load] taxonomy: {taxonomy_xlsx}"))
    xls = pd.ExcelFile(taxonomy_xlsx)
    taxonomy_df = pd.read_excel(xls, "taxonomy")
    rules_df = pd.read_excel(xls, "rules") if "rules" in xls.sheet_names else None
    exact_df = pd.read_excel(xls, "exact_map") if "exact_map" in xls.sheet_names else None

    print(_c(INFO, f"[rules] using examples column: '{examples_col}'"))
    taxonomy, rules, exact = load_taxonomy_from_frames(
        taxonomy_df,
        rules_df=rules_df,
        exact_map_df=exact_df,
        examples_col=examples_col,
    )

    # keyword map from the same examples column
    kw_by_l1 = keywords_by_l1_from_examples(taxonomy_df, examples_col=examples_col)

    l1_count = len(taxonomy)
    l2_count = sum(len(v) for v in taxonomy.values())
    print(_c(OK, f"[load] taxonomy: {l1_count} L1s / {l2_count} L2s; rules={len(rules)}; exact={len(exact)}"))

    # prompt-hints (optional, but call retained for completeness)
    extract_descriptions_from_taxonomy_df(taxonomy_df, examples_col=examples_col)

    # --- Batched mapping ---
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY not set for batched mapping.")
    print(_c(INFO, f"[batch] starting mapping (model={model}, temp={temperature}, min_confidence={min_confidence})"))

    mapped = batch_map_employees(
        employees_df=unique_roles,
        title_col=title_col,
        bu_col=bu_col,
        taxonomy=taxonomy,
        rules=rules,
        exact=exact,
        min_confidence=min_confidence,
        kw_by_l1=kw_by_l1,
        batch_size=int(os.getenv("BATCH_SIZE", "16")),
        model=model,
        temperature=temperature,
        progress=True,
        progress_label="[progress]",
        use_nc_l2_gpt=False,   # flip to True if you want GPT for NC L2s too
    )

    # --- optional: telemetry for reclassifications (titles only) ---
    if "Reclassified_L1" in mapped.columns:
        mask = mapped["Reclassified_L1"].notna()
        n_low = int(mask.sum())
        print(_c(WARN, f"[reclass] low-confidence rows: {n_low}"))
        if n_low:
            print(_c(INFO, "[reclass] titles (up to 20, lowest confidence first):"))
            titles = (
                mapped.loc[mask]
                      .sort_values(by=["Confidence"], ascending=True, na_position="first")[title_col]
                      .head(20)
            )
            for t in titles:
                print(f" - {t}")

    # quick telemetry
    try:
        print(_c(OK, "[done] MapSource distribution:"))
        print(mapped["MapSource"].value_counts(dropna=False).to_string())
        if "Mapped_L1" in mapped.columns:
            print(_c(OK, "[done] Sample L1/L2:"))
            print(mapped[["Mapped_L1", "Mapped_L2"]].head().to_string(index=False))
    except Exception:
        pass

    return mapped, employees_df


if __name__ == "__main__":
    CENSUS_PATH   = os.getenv("CENSUS_PATH", "input_census.csv")
    TAXONOMY_PATH = os.getenv("TAXONOMY_PATH", "taxonomy_updated.xlsx")
    TITLE_COL     = os.getenv("TITLE_COL", "business_title")
    BU_COL        = os.getenv("BU_COL", "job_family_group")
    OUTPUT        = os.getenv("OUTPUT", "mapped_census.csv")
    MODEL         = os.getenv("OPENAI_MODEL", "gpt-4o")
    TEMP          = float(os.getenv("OPENAI_TEMPERATURE", "0.0"))
    VERBOSE       = _to_bool(os.getenv("VERBOSE"), default=True)
    MIN_CONF      = float(os.getenv("MIN_CONFIDENCE", "0.8"))
    # NOTE: examples column in the taxonomy workbook; your new logic expects "Final Titles"
    EXAMPLES_COL  = os.getenv("EXAMPLES_COL", "Final Titles")

    df, employee_df = run_mapping_with_files(
        census_path=CENSUS_PATH,
        taxonomy_xlsx=TAXONOMY_PATH,
        title_col=TITLE_COL,
        bu_col=BU_COL,
        use_ai_fallback=True,
        model=MODEL,
        temperature=TEMP,
        verbose=VERBOSE,
        min_confidence=MIN_CONF,
        examples_col=EXAMPLES_COL,
    )

    Complete_mapped_census  = pd.merge(
        employee_df,
        df,
        on=[TITLE_COL, BU_COL],
        how='left'
    )

    Commercial_functions = [
        "Sales",
        "Marketing",
        "Customer Success",
        "Customer Support",
        "Pricing",
        "Revenue Operations (RevOps)",
        "Product Management",
    ]

    Complete_mapped_census['Updated_Role type (Commercial/ Non-Commercial)'] = (
        Complete_mapped_census['Mapped_L1']
        .apply(lambda g: "Commercial" if isinstance(g, str) and g.strip() in Commercial_functions
                         else "Non-commercial")
    )

    role_col = "Updated_Role type (Commercial/ Non-Commercial)"
    mask = Complete_mapped_census[role_col].fillna("").str.strip().eq("Non-commercial")

    vals = Complete_mapped_census.loc[mask, BU_COL].to_numpy()
    Complete_mapped_census.loc[mask, ["Mapped_L1", "Mapped_L2"]] = np.c_[vals, vals]

    Complete_mapped_census = Complete_mapped_census.reset_index(drop=True)

    df_with_paths, sup_cols = build_supervisor_paths(
        Complete_mapped_census,
        emp_col="employee_id",
        sup_col="supervisor_id",
        role_col="Updated_Role type (Commercial/ Non-Commercial)",
        l1_col="Mapped_L1",
    )

    df_final = mark_role_type_match(
        df_with_paths,
        path_col="Path_RoleType",
        target_col="Updated_Role type (Commercial/ Non-Commercial)",
        n=4,
        out_col="No_hierarchy_Shift?"
    )

    df_final.to_csv(OUTPUT, index=False)
