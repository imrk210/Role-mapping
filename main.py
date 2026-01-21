#!/usr/bin/env python3
from __future__ import annotations

import csv
import os
import pandas as pd

from dotenv import load_dotenv
load_dotenv()

from Functions.org_paths import build_emp_all_paths
from Functions.ic_manager_classifier import classify_ic_manager
from Functions.overwrite_roles import write_mapping_to_copy

from Functions.batch_mapping import (
    load_taxonomy_from_frames,
    batch_map_l1_only,
    batch_map_l2_from_l1,
)

from Functions.reclassify_l1_on_reportee_function import reclassify_l1_using_descendants_and_supervisors


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

ROLE_COL = "Updated_Role type (Commercial/ Non-Commercial)"

INFO = "36"
OK = "32"


def _c(c: str, s: str) -> str:
    return f"\033[{c}m{s}\033[0m"


def _sanitize_df_for_csv(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    # Prevent Excel/CSV row-breaking on embedded CR/LF in long path cells.
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = out[c].astype(str).str.replace(r"[\r\n]+", " ", regex=True)
    return out


def _sanitize_series_for_csv(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
        .str.replace("\r\n", " ", regex=False)
        .str.replace("\n", " ", regex=False)
        .str.replace("\r", " ", regex=False)
    )


def run_mapping_with_files(
    census_path: str,
    taxonomy_xlsx: str,
    *,
    title_col: str,
    bu_col: str,
    emp_id_col: str,
    sup_id_col: str,
    model: str = "gpt-4o",
    temperature: float = 0.0,
    l1_reclass_threshold: float = 0.8,
    examples_col: str = "Final Titles",
    batch_size: int = 25,
) -> pd.DataFrame:
    print(_c(INFO, f"[load] census: {census_path}"))

    if census_path.lower().endswith((".xlsx", ".xls", ".xlsm")):
        employees_df = pd.read_excel(
            census_path,
            sheet_name="Employee census",
            skiprows=12,
            header=0,
            engine="openpyxl",
        )
        if employees_df.shape[1] > 1:
            employees_df = employees_df.iloc[:, 1:].copy()
    else:
        employees_df = pd.read_csv(
            census_path,
            skiprows=12,
            header=0,
            sep=None,
            engine="python",
            encoding="utf-8-sig",
            on_bad_lines="skip",
        )
        if employees_df.shape[1] > 1:
            employees_df = employees_df.iloc[:, 1:].copy()

    # column guards
    for col in (title_col, bu_col, emp_id_col, sup_id_col):
        if col not in employees_df.columns:
            raise KeyError(f"Missing required column: {col!r}")

    # compute Span before mapping (safe even if many blanks)
    span_counts = employees_df[sup_id_col].value_counts(dropna=True)
    employees_df["Span"] = employees_df[emp_id_col].map(span_counts).fillna(0).astype(int)

    print(_c(OK, f"[load] census rows: {len(employees_df)}"))

    # taxonomy workbook
    print(_c(INFO, f"[load] taxonomy: {taxonomy_xlsx}"))
    xls = pd.ExcelFile(taxonomy_xlsx)
    taxonomy_df = pd.read_excel(xls, "taxonomy")
    rules_df = pd.read_excel(xls, "rules") if "rules" in xls.sheet_names else None
    exact_df = pd.read_excel(xls, "exact_map") if "exact_map" in xls.sheet_names else None

    taxonomy, rules, _ = load_taxonomy_from_frames(
        taxonomy_df,
        rules_df=rules_df,
        exact_map_df=exact_df,
        examples_col=examples_col,
    )

    # L1 mapping on unique roles
    unique_roles = employees_df[[title_col, bu_col]].drop_duplicates().sort_values(by=title_col)
    print(_c(OK, f"[prep] unique roles for L1: {len(unique_roles)}"))

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY not set.")

    print(_c(INFO, f"[L1] mapping (model={model}, temp={temperature})"))
    role_l1 = batch_map_l1_only(
        unique_roles,
        title_col=title_col,
        bu_col=bu_col,
        taxonomy_df=taxonomy_df,
        rules=rules,
        model=model,
        temperature=temperature,
        batch_size=batch_size,
        progress=True,
    )

    # merge back
    df = employees_df.merge(role_l1, on=[title_col, bu_col], how="left", validate="m:1")

    # role type derived from L1 value
    df[ROLE_COL] = df["Mapped_L1"].apply(
        lambda x: "Commercial" if isinstance(x, str) and x.strip() in CANON_SET else "Non-commercial"
    )

    # ------------------------------------------------------------------
    # Org-paths + L1 reclassify
    # FIX: if Manager ID blank/null => do NOT build paths nor reclassify for that row
    # ------------------------------------------------------------------
    sup_series = df[sup_id_col].copy()
    sup_str = sup_series.astype(str).str.strip()
    sup_missing = sup_series.isna() | sup_str.eq("") | sup_str.str.lower().isin(["nan", "none"])

    df["_skip_org_logic_"] = sup_missing

    # default path columns (keeps downstream stable)
    df["AllPath_IDs"] = ""
    df["AllPath_L1Paths"] = ""
    df["Is_in_layer_1_3"] = False
    df["Is_in_last_layer"] = False
    df["Is_Leaf"] = False

    # default reclass columns
    df["Reclassified_L1"] = ""
    df["Final_Mapped_L1"] = df["Mapped_L1"]

    df_org = df.loc[~df["_skip_org_logic_"]].copy()
    if len(df_org) > 0:
        print(_c(INFO, "[paths] building all paths (IDs + L1 paths) for rows with Manager ID"))
        df_org = build_emp_all_paths(
            df_org,
            emp_col=emp_id_col,
            sup_col=sup_id_col,
            l1_col="Mapped_L1",
            sep=" | ",
            path_sep=" || ",
        )

        required = ["AllPath_IDs", "AllPath_L1Paths", "Is_in_layer_1_3", "Is_in_last_layer", "Is_Leaf"]
        missing = [c for c in required if c not in df_org.columns]
        if missing:
            raise KeyError(f"build_emp_all_paths missing columns: {missing}")

        df.loc[df_org.index, required] = df_org[required]

        print(_c(INFO, f"[reclass] L1 reclass (descendants + supervisor for bottommost), threshold={l1_reclass_threshold}"))
        df_org = reclassify_l1_using_descendants_and_supervisors(
            df_org,
            emp_col=emp_id_col,
            sup_col=sup_id_col,
            l1_col="Mapped_L1",
            l1_conf_col="L1_Confidence",
            allpath_ids_col="AllPath_IDs",
            is_layer_1_3_col="Is_in_layer_1_3",
            is_last_layer_col="Is_in_last_layer",
            threshold=l1_reclass_threshold,
            out_reclass_col="Reclassified_L1",
            out_final_col="Final_Mapped_L1",
            min_votes=2,
            max_sup_layers=4,
            sep=" | ",
            path_sep=" || ",
        )

        df.loc[df_org.index, ["Reclassified_L1", "Final_Mapped_L1"]] = df_org[["Reclassified_L1", "Final_Mapped_L1"]]

    df.drop(columns=["_skip_org_logic_"], inplace=True, errors="ignore")

    # finalize role type AFTER final L1
    df[ROLE_COL] = df["Final_Mapped_L1"].apply(
        lambda x: "Commercial" if isinstance(x, str) and x.strip() in CANON_SET else "Non-commercial"
    )

    # L2 mapping
    print(_c(INFO, "[L2] mapping from Final_Mapped_L1"))
    l2_keys = df[[title_col, bu_col, "Final_Mapped_L1"]].drop_duplicates()

    l2_mapped = batch_map_l2_from_l1(
        l2_keys,
        title_col=title_col,
        bu_col=bu_col,
        final_l1_col="Final_Mapped_L1",
        taxonomy=taxonomy,
        taxonomy_df=taxonomy_df,
        rules=rules,
        model=model,
        temperature=temperature,
        batch_size=batch_size,
        progress=True,
    )

    df = df.merge(
        l2_mapped[[title_col, bu_col, "Final_Mapped_L1", "Mapped_L2", "L2_Confidence", "L2_Source"]],
        on=[title_col, bu_col, "Final_Mapped_L1"],
        how="left",
        validate="m:1",
    )

    # IC/Manager tagging
    ic_col = os.getenv("IC_MAN_COL", "IC/Manager")
    if ic_col not in df.columns or not df[ic_col].notna().any():
        df[ic_col] = df.apply(classify_ic_manager, axis=1)

    # Manager suffix column ONLY (do not overwrite Mapped_L2)
    df["Mapped_L2_to_write"] = df["Mapped_L2"].astype(str)
    mask_mgr = (df[ROLE_COL] == "Commercial") & (df[ic_col].astype(str).str.strip() == "Manager")
    df.loc[mask_mgr, "Mapped_L2_to_write"] = df.loc[mask_mgr, "Mapped_L2_to_write"] + " - Manager"

    # sanitize path columns to stop Excel row-splitting
    df = _sanitize_df_for_csv(df, ["AllPath_IDs", "AllPath_L1Paths"])

    # output only requested working columns
    keep_cols = [
        emp_id_col,
        title_col,
        bu_col,
        sup_id_col,
        "Mapped_L1",
        "L1_Confidence",
        "AllPath_IDs",
        "AllPath_L1Paths",
        "Is_in_layer_1_3",
        "Is_in_last_layer",
        "Reclassified_L1",
        "Final_Mapped_L1",
        ROLE_COL,
        "Mapped_L2",
        "L2_Confidence",
        ic_col,
        "Mapped_L2_to_write",
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols].copy()

    return df


if __name__ == "__main__":
    CENSUS_PATH = os.getenv("CENSUS_PATH", "Files/Input/PaCE_Input Format_updated.xlsm")
    TAXONOMY_PATH = os.getenv("TAXONOMY_PATH", "Files/Taxonomy/New_taxonomy.xlsx")
    TITLE_COL = os.getenv("TITLE_COL", "Job-Profile")
    BU_COL = os.getenv("BU_COL", "Team")
    EMP_ID_COL = os.getenv("EMP_ID_COL", "Employee ID")
    SUP_ID_COL = os.getenv("SUP_ID_COL", "Manager ID")
    IC_MAN_COL = os.getenv("IC_MAN_COL", "IC/Manager")
    OUTPUT = os.getenv("OUTPUT", "Files/Output/Mapped_census_working_file.csv")
    BATCH_SIZE = int(os.getenv("BATCH_SIZE", "25"))
    MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
    TEMP = float(os.getenv("OPENAI_TEMPERATURE", "0.0"))
    EXAMPLES_COL = os.getenv("EXAMPLES_COL", "Example Titles (Additional)")
    L1_RECLASS_TH = float(os.getenv("L1_RECLASS_THRESHOLD", "0.8"))

    df_final = run_mapping_with_files(
        census_path=CENSUS_PATH,
        taxonomy_xlsx=TAXONOMY_PATH,
        title_col=TITLE_COL,
        bu_col=BU_COL,
        emp_id_col=EMP_ID_COL,
        sup_id_col=SUP_ID_COL,
        model=MODEL,
        temperature=TEMP,
        l1_reclass_threshold=L1_RECLASS_TH,
        examples_col=EXAMPLES_COL,
        batch_size=BATCH_SIZE,
    )

    # sanitize path cols again before CSV write (extra guard)
    for c in ["AllPath_IDs", "AllPath_L1Paths"]:
        if c in df_final.columns:
            df_final[c] = _sanitize_series_for_csv(df_final[c])

    os.makedirs(os.path.dirname(OUTPUT) or ".", exist_ok=True)

    df_final.to_csv(
        OUTPUT,
        index=False,
        quoting=csv.QUOTE_ALL,
        escapechar="\\",
    )

    # Writeback:
    # overwrite Team in input file with Final_Mapped_L1
    # overwrite GTM roles mapped with Mapped_L2_to_write
    output_folder = os.getenv("OUTPUT_FOLDER", "Files/Output")
    output_file, n_updated, n_skipped = write_mapping_to_copy(
        input_xlsm_path=CENSUS_PATH,
        output_folder=output_folder,
        df_final=df_final,
        sheet_name="Employee census",
        header_row=13,
        emp_id_col=EMP_ID_COL,
        sup_id_col=SUP_ID_COL,
        title_col=TITLE_COL,
        team_col=BU_COL,
        gtm_col="GTM roles mapped",
        final_l1_col="Final_Mapped_L1",
        l2_write_col="Mapped_L2_to_write",
        l1_conf_col="L1_Confidence",  # ✅ NEW: writes to "Mapped_L1_Confidence" column in the output file
        # Optional (defaults shown for clarity):
        # out_l1_header="Mapped_L1",
        # out_l1_conf_header="Mapped_L1_Confidence",
        # out_l2_header="Mapped_L2",
    )



    print(f"Created file: {output_file}")
    print(f"Updated rows: {n_updated}")
    print(f"Skipped rows (no usable key): {n_skipped}")
