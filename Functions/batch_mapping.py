# batch_mapping.py
# ----------------------------- BATCHED MAPPER + TAXONOMY HELPERS -----------------------------
from __future__ import annotations
import numpy as np  # <-- add this (or add once at file top)
import json
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from openai import OpenAI

from Functions.service import Rule  # your existing Rule dataclass

__all__ = [
    "load_taxonomy_from_frames",
    "keywords_by_l1_from_examples",
    "batch_map_l1_only",
    "batch_map_l2_from_l1",
    "batch_map_employees",  # compatibility stub
]

# =============================================================================
# CONSTANTS
# =============================================================================

CANONICAL_7: List[str] = [
    "Sales",
    "Marketing",
    "Customer Success",
    "Customer Support",
    "Pricing",
    "Revenue Operations (RevOps)",
    "Product Management",
]
_CANON_SET = set(CANONICAL_7)

# =============================================================================
# SMALL UTILS
# =============================================================================

def _ensure_columns(df: pd.DataFrame, required: List[str], name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")

def _dedup_in_order(seq: List[str]) -> List[str]:
    return list(dict.fromkeys([x for x in seq if x]))

def _chunk_iter(seq: List[Any], n: int):
    for i in range(0, len(seq), n):
        yield seq[i : i + n]

def _print_progress(processed: int, total: int, *, label: str) -> None:
    if total <= 0:
        return
    pct = 100.0 * processed / total
    print(f"\r{label} {processed}/{total} ({pct:.1f}%)", end="", flush=True)

def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default

def _retry_chat_completion(
    client: OpenAI,
    *,
    model: str,
    temperature: float,
    messages: List[dict],
    max_retries: int = 4,
    retry_sleep_s: float = 1.2,
) -> str:
    last_err: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                temperature=temperature,
                messages=messages,
            )
            content = (resp.choices[0].message.content or "").strip()
            if not content:
                raise ValueError("empty model content")
            return content
        except Exception as e:
            last_err = e
            if attempt < max_retries:
                time.sleep(retry_sleep_s * attempt)
                continue
            raise RuntimeError(f"OpenAI call failed after {max_retries} attempts: {e}") from e
    raise RuntimeError(f"OpenAI call failed: {last_err}")

def _extract_json_block_safe(text: str) -> Any:
    if not text:
        raise ValueError("empty model reply")
    t = text.strip()

    t = t.replace("```json", "```").replace("```JSON", "```")
    if t.startswith("```"):
        t = t.strip("` \n")

    start_candidates = [(t.find("["), "["), (t.find("{"), "{")]
    start_candidates = [(i, ch) for i, ch in start_candidates if i != -1]
    if not start_candidates:
        raise ValueError("no JSON start found")

    start_idx, start_ch = min(start_candidates, key=lambda x: x[0])
    end_ch = "]" if start_ch == "[" else "}"
    end_idx = t.rfind(end_ch)
    if end_idx == -1 or end_idx <= start_idx:
        raise ValueError("no JSON end found")

    blob = t[start_idx : end_idx + 1]
    return json.loads(blob)

def _coerce_list_of_dicts(x: Any) -> List[dict]:
    if isinstance(x, list):
        return [d for d in x if isinstance(d, dict)]
    if isinstance(x, dict):
        return [x]
    return []

# =============================================================================
# TOKENIZATION + EXAMPLES
# =============================================================================

_STOPWORDS = {
    "and",
    "&",
    "of",
    "the",
    "global",
    "intl",
    "sr",
    "senior",
    "jr",
    "associate",
    "assistant",
}

_CANON_MAP = {
    "mktg": "marketing",
    "mar": "marketing",
    "acct": "account",
    "pm": "product",
    "ops": "operations",
}

def _tokenize(s: str) -> List[str]:
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9\s]+", " ", s)
    toks = [_CANON_MAP.get(w, w) for w in s.split() if w]
    bigrams = [" ".join(pair) for pair in zip(toks, toks[1:])]
    return toks + bigrams

def _split_examples(cell: Any) -> List[str]:
    if cell is None:
        return []
    s = str(cell).strip()
    if not s or s.lower() == "nan":
        return []
    return [t.strip() for t in s.split(",") if t.strip()]

def _build_examples_for_l1(
    taxonomy_df: pd.DataFrame,
    *,
    examples_col: str = "Example Titles (Additional)",
    max_per_l1: int = 18,
) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {k: [] for k in CANONICAL_7}
    if examples_col not in taxonomy_df.columns:
        return out

    for _, row in taxonomy_df.fillna("").iterrows():
        l1 = str(row.get("L1", "")).strip()
        if l1 not in _CANON_SET:
            continue
        exs = _split_examples(row.get(examples_col, ""))
        for ex in exs:
            if ex and len(out[l1]) < max_per_l1:
                out[l1].append(ex)
    return out

# =============================================================================
# TAXONOMY LOADER
# =============================================================================

def _phrase_to_regex(phrase: str) -> str:
    tokens = re.findall(r"[a-zA-Z]+", str(phrase).lower())
    parts = []
    for tok in tokens:
        if tok in _STOPWORDS:
            continue
        parts.append(fr"(?=.*\b{re.escape(tok)}\b)")
    return "".join(parts) + ".*" if parts else r"$^"

def load_taxonomy_from_frames(
    taxonomy_df: pd.DataFrame,
    rules_df: Optional[pd.DataFrame] = None,
    exact_map_df: Optional[pd.DataFrame] = None,
    *,
    examples_col: str = "Example Titles (Additional)",
    auto_rule_priority: int = 90,
) -> Tuple[Dict[str, List[str]], List[Rule], Dict[Tuple[str, Optional[str]], Tuple[str, str]]]:
    taxonomy_df = taxonomy_df.fillna("")
    _ensure_columns(taxonomy_df, ["L1", "L2"], "taxonomy_df")

    tdf = taxonomy_df[["L1", "L2"]].astype(str).copy()
    tdf["L1"] = tdf["L1"].str.strip()
    tdf["L2"] = tdf["L2"].str.strip()

    taxonomy: Dict[str, List[str]] = {
        l1: _dedup_in_order(grp["L2"].tolist())
        for l1, grp in tdf.groupby("L1", sort=False)
        if l1
    }

    rules: List[Rule] = []

    if rules_df is not None:
        r = rules_df.fillna("")
        _ensure_columns(r, ["L1", "L2", "pattern", "match_type"], "rules_df")
        for _, row in r.iterrows():
            rules.append(
                Rule(
                    l1=str(row["L1"]).strip(),
                    l2=str(row["L2"]).strip(),
                    pattern=str(row["pattern"]).strip(),
                    match_type=str(row["match_type"]).strip().lower(),
                    bu=(str(row["bu"]).strip() or None) if "bu" in r.columns else None,
                    cost_center=None,
                    priority=int(row["priority"])
                    if ("priority" in r.columns and str(row["priority"]).strip())
                    else 100,
                    notes=(str(row["notes"]).strip() or None) if "notes" in r.columns else None,
                )
            )

    if examples_col in taxonomy_df.columns:
        for _, row in taxonomy_df.iterrows():
            l1 = str(row["L1"]).strip()
            l2 = str(row["L2"]).strip()
            if not (l1 and l2):
                continue
            examples = _split_examples(row[examples_col])
            for ex in examples:
                pat = _phrase_to_regex(ex)
                if pat == r"$^":
                    continue
                rules.append(
                    Rule(
                        l1=l1,
                        l2=l2,
                        pattern=pat,
                        match_type="regex",
                        bu=None,
                        cost_center=None,
                        priority=auto_rule_priority,
                        notes=f"auto from {examples_col}: {ex}",
                    )
                )

    return taxonomy, rules, {}

def keywords_by_l1_from_examples(
    taxonomy_df: pd.DataFrame,
    *,
    examples_col: str = "Example Titles (Additional)",
) -> Dict[str, Dict[str, List[str]]]:
    out: Dict[str, Dict[str, List[str]]] = {}
    if examples_col not in taxonomy_df.columns:
        return out

    for _, r in taxonomy_df.fillna("").iterrows():
        l1 = str(r.get("L1", "")).strip()
        l2 = str(r.get("L2", "")).strip()
        if not (l1 and l2):
            continue
        examples = _split_examples(r.get(examples_col, ""))
        tokset: set[str] = set()
        for ex in examples:
            tokset.update(_tokenize(ex))
        if tokset:
            out.setdefault(l1, {}).setdefault(l2, []).extend(sorted(tokset))

    for l1 in out:
        for l2 in out[l1]:
            out[l1][l2] = sorted(set(out[l1][l2]))
    return out

# =============================================================================
# BU NORMALIZATION (COMMERCIAL ONLY)
# =============================================================================

_BU_SYNONYMS = {
    "sales": "Sales",
    "marketing": "Marketing",
    "customer success": "Customer Success",
    "cs": "Customer Success",
    "customer support": "Customer Support",
    "support": "Customer Support",
    "pricing": "Pricing",
    "revenue operations": "Revenue Operations (RevOps)",
    "rev ops": "Revenue Operations (RevOps)",
    "revops": "Revenue Operations (RevOps)",
    "revenue ops": "Revenue Operations (RevOps)",
    "product management": "Product Management",
    "pm": "Product Management",
    "prod mgmt": "Product Management",
}

def _normalize_bucket(bu: Optional[str]) -> Optional[str]:
    if not bu:
        return None
    s = re.sub(r"[^a-z0-9\s/()-]", " ", str(bu).lower()).strip()
    s = re.sub(r"\s+", " ", s)
    for c in _CANON_SET:
        if s == c.lower():
            return c
    return _BU_SYNONYMS.get(s)

# =============================================================================
# SHARED LEADERSHIP POLICY (USED BY L1 + L2)
# =============================================================================

_MANAGER_RE = re.compile(r"\b(manager|mgr|team\s*lead|teamlead|lead)\b", re.I)

_TRUE_LEADERSHIP_RE = re.compile(
    r"\b("
    r"ceo|cfo|coo|cio|cto|cmo|cro|cso|chro|cpo|"
    r"chief\s+\w+(?:\s+\w+){0,2}\s+officer|"
    r"president|"
    r"(executive\s+)?vice\s+president|\b(evp|svp|avp|rvp|vp)\b|"
    r"\bdirector\b|\bdir\b|"
    r"\bhead(\s+of)?\b"
    r")\b",
    re.I,
)

def _has_manager_signal(title: str) -> bool:
    return bool(title and _MANAGER_RE.search(title))

def _is_true_leadership_title(title: str) -> bool:
    if not title:
        return False
    if _has_manager_signal(title):
        return False
    return bool(_TRUE_LEADERSHIP_RE.search(title))

# =============================================================================
# RULES MATCHING (HINTS ONLY)
# =============================================================================

def _match_rule(job_title: str, rule: Rule, bu_raw: Optional[str]) -> bool:
    if rule.bu:
        bu_norm = _normalize_bucket(bu_raw)
        if bu_norm != rule.bu:
            return False

    jt = job_title or ""

    if rule.match_type == "regex":
        try:
            comp = getattr(rule, "_compiled", None)
            if comp is None or comp[0] != rule.pattern:
                comp = (rule.pattern, re.compile(rule.pattern, flags=re.IGNORECASE))
                setattr(rule, "_compiled", comp)
            return comp[1].search(jt) is not None
        except re.error:
            return False

    if rule.match_type == "contains":
        return rule.pattern.lower() in jt.lower()

    if rule.match_type == "equals":
        return jt.strip().lower() == rule.pattern.strip().lower()

    return False

def _rule_hints_for_title(
    rules: List[Rule],
    *,
    title: str,
    bu_raw: Optional[str],
    stage: str,
    max_hints: int = 6,
) -> List[dict]:
    hits: List[Tuple[int, int, str, str]] = []
    for r in (rules or []):
        if stage == "L1" and r.l1 not in _CANON_SET:
            continue
        if _match_rule(title, r, bu_raw):
            hits.append((int(r.priority), len(r.pattern or ""), r.l1, r.l2))

    hits.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return [{"l1": l1, "l2": l2, "priority": pr} for pr, _, l1, l2 in hits[:max_hints]]

# =============================================================================
# TAXONOMY-DERIVED COMMERCIAL C-SUITE -> COMMERCIAL L1 OVERRIDE (NO HARDCODE)
# =============================================================================

_CS_GATED = re.compile(r"\b(chief|ceo|cfo|coo|cio|cto|cmo|cro|cso|chro|cpo)\b", re.I)
_PARENS_RE = re.compile(r"\([^)]*\)")

def _build_commercial_csuite_rules_from_taxonomy(
    taxonomy_df: pd.DataFrame,
    *,
    examples_col: str = "Example Titles (Additional)",
) -> List[Tuple[str, re.Pattern]]:
    """
    Build [(commercial_l1, compiled_regex)] from taxonomy example titles that look C-suite.

    Critical fix:
    - Example titles often contain acronyms in parentheses: "Chief Technology Officer (CTO)"
      Your employee titles often omit the "(CTO)" part.
      If we include "(CTO)" as required tokens, matches fail.
    - So we build patterns from the example with parentheses stripped.
    """
    if examples_col not in taxonomy_df.columns:
        return []

    out: List[Tuple[str, re.Pattern]] = []
    tdf = taxonomy_df.fillna("")

    for _, row in tdf.iterrows():
        l1 = str(row.get("L1", "")).strip()
        if l1 not in _CANON_SET:
            continue

        examples = _split_examples(row.get(examples_col, ""))
        for ex in examples:
            exs = str(ex).strip()
            if not exs or not _CS_GATED.search(exs):
                continue

            # ✅ strip "(CTO)" / "(CRO)" / etc so they don't become required tokens
            exs_clean = _PARENS_RE.sub(" ", exs)
            exs_clean = re.sub(r"\s+", " ", exs_clean).strip()
            if not exs_clean:
                continue

            pat = _phrase_to_regex(exs_clean)
            if pat == r"$^":
                continue

            try:
                out.append((l1, re.compile(pat, flags=re.I)))
            except re.error:
                continue

    seen = set()
    deduped: List[Tuple[str, re.Pattern]] = []
    for l1, rx in out:
        k = (l1, rx.pattern)
        if k in seen:
            continue
        seen.add(k)
        deduped.append((l1, rx))

    return deduped

def _match_any_csuite_rule(title: str, csuite_rules: List[Tuple[str, re.Pattern]]) -> Optional[str]:
    if not title:
        return None
    for l1, rx in csuite_rules:
        if rx.search(title):
            return l1
    return None

# =============================================================================
# COMMERCIAL ANCHORS (must/contra/fallbacks) + REVOPS STRATEGY HINT
# =============================================================================

ANCHORS: Dict[str, Dict[str, Any]] = {
    "Sales": {
        "must": r"\b("
        r"account\s+executive|\bae\b|"
        r"field\s+sales|inside\s+sales|"
        r"territory\s+sales|regional\s+sales|"
        r"sales\s+(rep|representative|specialist)(?!\s*operations)|"
        r"quota|hunter"
        r")\b",
        "contra": r"\b("
        r"proposal|rfp|rfi|tender|bid|"
        r"enablement|revenue\s+enablement|sales\s+enablement|"
        r"operations|ops|analytics|"
        r"event|events|marketing|"
        r"administrative|assistant"
        r")\b",
        "fallbacks": [
            ("Customer Success", r"\b(renewal|renewals|retention|account\s+management|customer\s+success|csm)\b"),
            (
                "Revenue Operations (RevOps)",
                r"\b(proposal|rfp|rfi|tender|bid|revenue\s+enablement|sales\s+enablement|enablement|sales\s*ops)\b",
            ),
            ("Marketing", r"\b(event|events|field\s+marketing|campaign|demand\s+gen)\b"),
        ],
    },
    "Marketing": {
        "must": r"\b(marketing|campaign|demand|brand|content|growth|events?)\b",
        "contra": r"\b(product\s+management|\bpm\b|pricing|deal\s*desk|revops|sales\s*ops)\b",
        "fallbacks": [("Product Management", r"\b(product\s+(marketing|manager|owner)|\bpm\b)\b")],
    },
    "Customer Success": {
        "must": r"\b(customer\s+success|csm|onboarding|adoption|health|renewal(s)?)\b",
        "contra": r"\b(help\s?desk|ticket|tier[ -]?[1-4]|escalation|troubleshoot)\b",
        "fallbacks": [("Customer Support", r"\b(support|help\s?desk|ticket|tier[ -]?[1-4]|escalation)\b")],
    },
    "Customer Support": {
        "must": r"\b(support|help\s?desk|ticket|escalation|troubleshoot|tier[ -]?[1-4])\b",
        "contra": r"\b(customer\s+success|csm|renewal|adoption|onboarding)\b",
        "fallbacks": [("Customer Success", r"\b(customer\s+success|csm|onboarding|adoption|renewal(s)?)\b")],
    },
    "Revenue Operations (RevOps)": {
        "must": r"\b("
        r"rev(\.|enue)?\s*ops|revops|gtm\s*ops|sales\s*ops|"
        r"crm|sfdc|"
        r"enablement|revenue\s+enablement|sales\s+enablement|"
        r"proposal|rfp|rfi|tender|bid|"
        r"funnel|forecast|pipeline|analytics|operations|ops"
        r"technical education|enablement|education"
        r")\b",
        "contra": r"\b(quota|hunter|account\s+executive|\bae\b)\b",
        "fallbacks": [
            ("Pricing", r"\b(pricing|price\s*ops|deal\s*desk|margin|cpq)\b"),
            ("Sales", r"\b(quota|hunter|account\s+executive|\bae\b)\b"),
        ],
    },
    "Pricing": {
        "must": r"\b("
        r"pricing|price\s*strategy|deal\s*desk|cpq|"
        r"configure[-\s]*price[-\s]*quote|"
        r"quote[-\s]*to[-\s]*cash|q2c|"
        r"discount(?:ing)?|margin|profitabilit(?:y|ies)|"
        r"rate\s*card|pricebook|list\s*price"
        r")\b",
        "contra": r"\b(customer\s+success|support|marketing\s+manager)\b",
        "fallbacks": [("Revenue Operations (RevOps)", r"\b(rev(\.|enue)?\s*ops|revops|sales\s*ops|enablement|operations)\b")],
    },
    "Product Management": {
        "must": r"\b(product(s)?|product\s+(manager|owner|management|marketing)|\bpm\b)\b",
        "contra": r"\b(marketing\s+ops|revops|sales\s+ops|deal\s*desk|pricing)\b",
        "fallbacks": [("Marketing", r"\b(marketing|campaign|demand|brand|portfolio)\b")],
    },
}

_ANCH_RE: Dict[str, Dict[str, Any]] = {}
for k, cfg in ANCHORS.items():
    _ANCH_RE[k] = {
        "must": re.compile(cfg["must"], re.I) if cfg.get("must") else None,
        "contra": re.compile(cfg["contra"], re.I) if cfg.get("contra") else None,
        "fallbacks": [(fb, re.compile(pat, re.I)) for fb, pat in cfg.get("fallbacks", [])],
    }

_REVOPS_STRATEGY_TERMS = {
    "gtm",
    "go-to-market",
    "go to market",
    "strategy",
    "strategic",
    "revenue operations",
    "rev ops",
    "revops",
    "sales operations",
    "marketing operations",
    "enablement",
    "revenue enablement",
    "sales enablement",
    "planning",
    "operations",
    "ops",
    "analytics",
    "insights",
    "forecast",
    "pipeline",
    "territory",
    "compensation",
    "crm",
    "rfp",
    "proposal",
    "rfis",
    "tender",
    "bid",
    "technical education",
    "sfdc",
    "data",
    "data ops",
    "data operations",
}

def _needs_revops_strategy_hint(title: str) -> bool:
    if not title:
        return False
    t = title.lower()
    return any(term in t for term in _REVOPS_STRATEGY_TERMS)

def commercial_anchor_diagnostics(text: str) -> Dict[str, Any]:
    t = text or ""
    must_hits: List[str] = []
    contra_hits: List[str] = []
    fallback_hits: List[dict] = []

    for l1, cfg in _ANCH_RE.items():
        if cfg["must"] and cfg["must"].search(t):
            must_hits.append(l1)
        if cfg["contra"] and cfg["contra"].search(t):
            contra_hits.append(l1)
            for fb_l1, fb_rx in cfg["fallbacks"]:
                if fb_rx.search(t):
                    fallback_hits.append({"from": l1, "to": fb_l1})

    must_hits = list(dict.fromkeys(must_hits))
    contra_hits = list(dict.fromkeys(contra_hits))

    uniq_fb = []
    seen = set()
    for x in fallback_hits:
        key = (x["from"], x["to"])
        if key not in seen:
            seen.add(key)
            uniq_fb.append(x)

    return {"must_hits": must_hits, "contra_hits": contra_hits, "fallback_hits": uniq_fb[:8]}

# =============================================================================
# NON-COMMERCIAL ANCHOR HINTS (UNCHANGED)
# =============================================================================

COMMERCIAL_GUARD_RE = re.compile(
    r"\b("
    r"sales|account\s+executive|\bae\b|"
    r"bdr|sdr|business\s+development|sales\s+development|"
    r"marketing|"
    r"customer\s+success|customer\s+support|"
    r"rev(\.|enue)?\s*ops|revops|sales\s*ops|gtm\s*ops|enablement|"
    r"product\s+management|\bpm\b|pricing|deal\s*desk"
    r")\b",
    re.I,
)

NC_ANCHOR_PATTERNS = {
    "Human Resources": r"\b(human\s*resources?|hr|people\s*ops|talent\s*(acquisition|management)|recruit(ing|ment)|payroll|benefits|comp(ensation)?)\b",
    "Finance / Accounting": r"\b(finance|financial|account(ing|ant)|accounts?\s*payable|accounts?\s*receivable|treasury|controller|audit|auditor|tax|fp&?a)\b",
    "Legal / Compliance": r"\b(legal|attorney|lawyer|counsel|paralegal|compliance|regulatory)\b",
    "Information Technology": r"\b(it|information\s*technology|systems?\s*admin|sysadmin|network|help\s*desk|desktop\s*support|infrastructure|devops|sre|database|dba|cyber|infosec|security\s*(analyst|engineer))\b",
    "Engineering / R&D": r"\b((?<!sales\s)engineer(ing)?|developer|software|hardware|firmware|scientist|research|r&d|lab|chemist|physicist|biologist|data\s*scientist|qa|quality\s*assurance)\b",
    "Admin / Office / Clerical": r"\b(administrative|admin|office|receptionist|cleri?cal|\bclerk\b|secretary|assistant|data\s*entry|mailroom)\b",
    "Executive / General Management": r"\b(ceo|chief\s+\w+|coo|cfo|cio|cto|cso|chro|president|board|chair|founder|general\s*manager|gm|managing\s*director)\b",
}
_NC_RES = {k: re.compile(v, re.I) for k, v in NC_ANCHOR_PATTERNS.items()}

def nc_candidates(title: str, bu_raw: Optional[str]) -> List[str]:
    t = f"{(bu_raw or '')} | {(title or '')}".lower()
    cands: List[str] = []
    for label, rx in _NC_RES.items():
        if label == "Executive / General Management":
            if rx.search(t) and not COMMERCIAL_GUARD_RE.search(t):
                cands.append(label)
        else:
            if rx.search(t):
                cands.append(label)
    return list(dict.fromkeys(cands))

# =============================================================================
# L1 MAPPER (COMMERCIAL-FIRST; GPT DECIDES USING TAXONOMY + ANCHORS + HINTS)
# =============================================================================

def batch_map_l1_only(
    employees_df: pd.DataFrame,
    *,
    title_col: str,
    bu_col: Optional[str],
    taxonomy_df: pd.DataFrame,
    rules: Optional[List[Rule]] = None,
    model: str = "gpt-4o",
    temperature: float = 0.0,
    batch_size: int = 20,
    progress: bool = True,
    examples_col: str = "Example Titles (Additional)",
) -> pd.DataFrame:


    if title_col not in employees_df.columns:
        raise KeyError(f"employees_df missing title column: {title_col}")
    has_bu = bool(bu_col) and (bu_col in employees_df.columns)

    out = employees_df.copy()
    out["Mapped_L1"] = ""
    # IMPORTANT: L1_Confidence is GPT-only now. Deterministic paths leave it as NaN.
    out["L1_Confidence"] = np.nan
    out["L1_Source"] = ""

    # ✅ IMPORTANT: build csuite rules from the taxonomy column that actually contains those examples.
    csuite_rules = _build_commercial_csuite_rules_from_taxonomy(
        taxonomy_df,
        examples_col="Example Titles (Additional)",
    )

    # ORIGINAL WEIGHT LOGIC
    n_buckets = 0
    if has_bu:
        present = {_normalize_bucket(v) for v in out[bu_col].dropna().astype(str)}
        present = {p for p in present if p in _CANON_SET}
        n_buckets = len(present)

    if n_buckets <= 4:
        w_title, w_bu = 0.50, 0.50
    elif n_buckets <= 6:
        w_title, w_bu = 0.30, 0.70
    else:
        w_title, w_bu = 0.0, 1.0

    ex_by_l1 = _build_examples_for_l1(taxonomy_df, examples_col=examples_col, max_per_l1=18)
    short_examples = {l1: ex_by_l1.get(l1, [])[:12] for l1 in CANONICAL_7}

    rules = rules or []

    items: List[dict] = []
    for idx, row in out.iterrows():
        title = str(row[title_col]).strip() if pd.notna(row[title_col]) else ""
        bu_raw = str(row[bu_col]).strip() if (has_bu and pd.notna(row[bu_col])) else ""
        bu_norm = _normalize_bucket(bu_raw) or ""
        text = f"{bu_raw} | {title}".strip(" |")

        csuite_l1 = _match_any_csuite_rule(title, csuite_rules)
        if csuite_l1:
            items.append(
                {
                    "id": int(idx),
                    "job_title": title,
                    "job_family_group": bu_raw,
                    "bu_norm": bu_norm,
                    "text": text,
                    "lock_bu": False,
                    "forced_l1": csuite_l1,
                    "nc_candidates": [],
                    "commercial_anchors": {},
                    "hint_revops": False,
                    "rule_hints": [],
                }
            )
            continue

        lock_bu = (w_bu == 1.0) and (bu_norm in _CANON_SET)

        items.append(
            {
                "id": int(idx),
                "job_title": title,
                "job_family_group": bu_raw,
                "bu_norm": bu_norm,
                "text": text,
                "lock_bu": lock_bu,
                "nc_candidates": nc_candidates(title, bu_raw)[:5],
                "commercial_anchors": commercial_anchor_diagnostics(text),
                "hint_revops": bool(_needs_revops_strategy_hint(title)),
                "rule_hints": _rule_hints_for_title(
                    rules, title=title, bu_raw=(bu_raw or None), stage="L1", max_hints=6
                ),
            }
        )

    client = OpenAI()

    # results: id -> (mapped_l1, gpt_conf_or_nan, source)
    results: Dict[int, Tuple[str, float, str]] = {}

    total = len(items)
    done = 0

    choices = CANONICAL_7 + ["__KEEP_BU__"]

    for chunk in _chunk_iter(items, batch_size):
        to_model = []
        for it in chunk:
            if it.get("forced_l1"):
                # Deterministic: confidence MUST NOT be assigned here
                results[it["id"]] = (it["forced_l1"], float("nan"), "L1(csuite-from-taxonomy)")
            elif it["lock_bu"]:
                # Deterministic: confidence MUST NOT be assigned here
                results[it["id"]] = (it["bu_norm"], float("nan"), "L1(bu-locked)")
            else:
                to_model.append(it)

        if to_model:
            payload = [
                {
                    "id": it["id"],
                    "job_title": it["job_title"],
                    "job_family_group": it["job_family_group"],
                    "bu_norm": it["bu_norm"],
                    "rule_hints": it["rule_hints"],
                    "nc_candidates": it["nc_candidates"],
                    "commercial_anchors": it["commercial_anchors"],
                    "hint_revops": it["hint_revops"],
                }
                for it in to_model
            ]

            prompt = f"""
You are mapping L1 with a WEIGHTED decision rule.

Output must be EXACTLY one of:
{choices}

Meaning:
- If NON-COMMERCIAL: output "__KEEP_BU__"
- If COMMERCIAL: output exactly ONE of these 7: {CANONICAL_7}

MANDATORY weighting (this is the base assumption):
- Use job_title weight = {w_title:.2f}
- Use job_family_group weight = {w_bu:.2f}
- If the signals conflict, prefer the higher-weighted signal.
- If weights are equal, job_title breaks ties.

Critical clarification:
- "Executive" / "Corporate" / broad groups in job_family_group are NOT sufficient to call NON-COMMERCIAL.
  If job_title is clearly commercial (Sales/Marketing/CS/Support/Pricing/RevOps/PM), choose a commercial bucket.

Commercial examples from taxonomy (fuzzy match allowed; examples are NOT exact requirements):
{json.dumps(short_examples, ensure_ascii=False)}

Evidence provided per item (hints, not strict rules):
- rule_hints: matched taxonomy rule patterns (if any)
- commercial_anchors: must/contra/fallback evidence
- nc_candidates: non-commercial hints (HR/Finance/IT/etc.)

Confidence rules (MANDATORY):
- confidence MUST be a probability estimate in the range [0,1]
- use at least TWO decimal places (e.g., 0.17, 0.63, 0.92)
- do NOT round to common buckets like 0, 0.5, 0.9, or 1
- lower confidence when signals conflict or are weak
- higher confidence only when signals strongly agree


Return ONLY JSON:
[{{"id": <id>, "l1": "<one of choices>", "confidence": 0-1}}]

Items:
{json.dumps(payload, ensure_ascii=False, indent=2)}
""".strip()

            content = _retry_chat_completion(
                client,
                model=model,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}],
            )

            raw = _extract_json_block_safe(content)
            data = _coerce_list_of_dicts(raw)
            by_id = {int(d["id"]): d for d in data if isinstance(d, dict) and "id" in d}

            for it in to_model:
                rid = it["id"]
                d = by_id.get(rid, {})

                l1 = str(d.get("l1") or "").strip()
                conf = _safe_float(d.get("confidence"), 0.0)
                conf = float(max(0.0, min(conf, 1.0)))  # GPT-only confidence

                if l1 == "__KEEP_BU__":
                    mapped = (it["job_family_group"] or "").strip() or "Unknown"
                    results[rid] = (mapped, conf, "L1(gpt->keep-bu)")
                elif l1 in _CANON_SET:
                    results[rid] = (l1, conf, "L1(gpt)")
                else:
                    # invalid output: still keep BU, but confidence remains GPT-provided
                    mapped = (it["job_family_group"] or "").strip() or "Unknown"
                    results[rid] = (mapped, conf, "L1(gpt-invalid->keep-bu)")

        done += len(chunk)
        if progress:
            _print_progress(done, total, label="[L1]")

    if progress:
        print()

    for it in items:
        rid = it["id"]
        mapped, gpt_conf, src = results.get(
            rid,
            ((it["job_family_group"] or "Unknown").strip(), float("nan"), "fallback"),
        )
        out.at[rid, "Mapped_L1"] = mapped
        # GPT-only: deterministic mappings keep NaN
        out.at[rid, "L1_Confidence"] = gpt_conf
        out.at[rid, "L1_Source"] = src

    return out

# =============================================================================
# L2 MAPPER (COMMERCIAL: GPT CHOOSES FROM taxonomy[L1]; NC: KEEP BU)
# =============================================================================

def batch_map_l2_from_l1(
    df: pd.DataFrame,
    *,
    title_col: str,
    bu_col: Optional[str],
    final_l1_col: str,
    taxonomy: Dict[str, List[str]],
    rules: List[Rule],
    kw_by_l1: Optional[Dict[str, Dict[str, List[str]]]] = None,
    taxonomy_df: Optional[pd.DataFrame] = None,
    batch_size: int = 20,
    model: str = "gpt-4o",
    temperature: float = 0.0,
    progress: bool = True,
    **_ignored_kwargs: Any,
) -> pd.DataFrame:
    if title_col not in df.columns:
        raise KeyError(f"missing {title_col}")
    if final_l1_col not in df.columns:
        raise KeyError(f"missing {final_l1_col}")

    has_bu = bool(bu_col) and (bu_col in df.columns)
    rules = rules or []

    out = df.copy()
    out["Mapped_L2"] = ""
    out["L2_Confidence"] = 0.0
    out["L2_Source"] = ""
    out["Manager_Title_Flag"] = False

    client = OpenAI()

    def _rule_hints_for_title_l2(
        rules_: List[Rule],
        *,
        title: str,
        bu_raw: Optional[str],
        max_hints: int = 8,
    ) -> List[dict]:
        hits: List[Tuple[int, int, str, str]] = []
        for r in (rules_ or []):
            try:
                if _match_rule(title, r, bu_raw):
                    hits.append((int(r.priority), len(r.pattern or ""), r.l1, r.l2))
            except Exception:
                continue

        hits.sort(key=lambda x: (x[0], x[1]), reverse=True)
        return [{"l1": l1, "l2": l2, "priority": pr} for pr, _, l1, l2 in hits[:max_hints]]

    need_ai: List[dict] = []

    for idx, row in out.iterrows():
        title = str(row[title_col]).strip() if pd.notna(row[title_col]) else ""
        bu_raw = str(row[bu_col]).strip() if has_bu and pd.notna(row[bu_col]) else ""
        l1 = str(row[final_l1_col] or "").strip()

        has_mgr = _has_manager_signal(title)
        out.at[idx, "Manager_Title_Flag"] = bool(has_mgr)

        if l1 not in _CANON_SET:
            out.at[idx, "Mapped_L2"] = (bu_raw or "").strip() or l1 or "Unknown"
            out.at[idx, "L2_Confidence"] = 1.0
            out.at[idx, "L2_Source"] = "non-commercial->keep-bu"
            continue

        allowed_l2 = taxonomy.get(l1, [])
        if not allowed_l2:
            out.at[idx, "Mapped_L2"] = ""
            out.at[idx, "L2_Confidence"] = 0.0
            out.at[idx, "L2_Source"] = "no-allowed-l2"
            continue

        leadership_label = next((x for x in allowed_l2 if str(x).strip().lower() == "leadership"), None)
        is_leader = _is_true_leadership_title(title)

        if is_leader and leadership_label:
            out.at[idx, "Mapped_L2"] = leadership_label
            out.at[idx, "L2_Confidence"] = 0.99
            out.at[idx, "L2_Source"] = "forced-leadership"
            continue

        allowed_for_model = allowed_l2
        leadership_removed = False
        if leadership_label and (not is_leader or has_mgr):
            allowed_for_model = [x for x in allowed_l2 if str(x).strip().lower() != "leadership"]
            leadership_removed = True
            if not allowed_for_model:
                allowed_for_model = allowed_l2

        rule_hints = _rule_hints_for_title_l2(rules, title=title, bu_raw=(bu_raw or None), max_hints=8)

        kw_hint_lines: List[str] = []
        if kw_by_l1:
            kw_map = kw_by_l1.get(l1, {}) or {}
            for l2c in allowed_for_model:
                toks = kw_map.get(l2c, [])
                if toks:
                    kw_hint_lines.append(f"{l2c}: {', '.join(toks[:10])}")

        need_ai.append(
            {
                "id": int(idx),
                "text": f"{bu_raw} | {title}".strip(" |"),
                "l1": l1,
                "allowed_l2": allowed_for_model,
                "rule_hints": rule_hints,
                "kw_hints": " | ".join(kw_hint_lines[:12]),
                "leadership_removed": leadership_removed,
            }
        )

    total_ai = len(need_ai)
    done_ai = 0

    for chunk in _chunk_iter(need_ai, batch_size):
        payload = [
            {
                "id": it["id"],
                "text": it["text"],
                "l1": it["l1"],
                "allowed_l2": it["allowed_l2"],
                "rule_hints": it["rule_hints"],
                "kw_hints": it["kw_hints"],
                "leadership_removed": it["leadership_removed"],
            }
            for it in chunk
        ]

        prompt = f"""
STAGE = L2 (Commercial)

For each item, choose EXACTLY ONE L2 from allowed_l2.

Hard rules:
- Do NOT invent categories.
- If leadership_removed=true, DO NOT return "Leadership".

Return ONLY JSON array:
[{{"id": <id>, "l2": "<one of allowed_l2>", "confidence": 0-1}}]

Items:
{json.dumps(payload, ensure_ascii=False, indent=2)}
""".strip()

        content = _retry_chat_completion(
            client,
            model=model,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = _extract_json_block_safe(content)
        data = _coerce_list_of_dicts(raw)
        by_id = {int(d["id"]): d for d in data if isinstance(d, dict) and "id" in d}

        for it in chunk:
            rid = it["id"]
            d = by_id.get(rid, {})

            l2 = str(d.get("l2", "")).strip()
            conf = _safe_float(d.get("confidence"), 0.0)

            allowed = it["allowed_l2"]

            if not l2 or l2 not in allowed:
                l2 = allowed[0] if allowed else ""
                conf = min(conf, 0.49)
                src = "ai-L2(invalid->fallback)"
            else:
                src = "ai-L2"

            if it["leadership_removed"] and str(l2).strip().lower() == "leadership":
                l2 = next(
                    (x for x in allowed if str(x).strip().lower() != "leadership"),
                    allowed[0] if allowed else "",
                )
                conf = min(conf, 0.49)
                src += "+leadership-blocked"

            out.at[rid, "Mapped_L2"] = l2
            out.at[rid, "L2_Confidence"] = float(max(0.0, min(conf, 1.0)))
            out.at[rid, "L2_Source"] = src

        done_ai += len(chunk)
        if progress:
            _print_progress(done_ai, total_ai or 1, label="[L2]")

    if progress:
        print()

    return out

# =============================================================================
# COMPATIBILITY STUB
# =============================================================================

def batch_map_employees(*args, **kwargs):
    raise NotImplementedError(
        "batch_map_employees() is not used anymore. Use batch_map_l1_only() then batch_map_l2_from_l1()."
    )
