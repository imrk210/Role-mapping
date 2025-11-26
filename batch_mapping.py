# batch_mapping.py
# ----------------------------- BATCHED MAPPER + TAXONOMY HELPERS -----------------------------
from __future__ import annotations
import json, re, math, os
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
from openai import OpenAI
from service import Rule, RoleTagger  # RoleTagger kept for symmetry; not required in batch_map_employees

__all__ = [
    "load_taxonomy_from_frames",
    "extract_descriptions_from_taxonomy_df",
    "keywords_by_l1_from_examples",
    "batch_map_employees",
]

# ---------------- helpers ----------------

def _ensure_columns(df: pd.DataFrame, required: List[str], name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")

def _dedup_in_order(seq: List[str]) -> List[str]:
    return list(dict.fromkeys([x for x in seq if x]))

def _split_examples(cell: Any) -> List[str]:
    if cell is None:
        return []
    s = str(cell).strip()
    if not s or s.lower() == "nan":
        return []
    return [t.strip() for t in s.split(",") if t.strip()]

# Canonical L1 buckets the taxonomy covers (commercial GTM functions)
_CANONICAL_BUCKETS = {
    "Sales",
    "Marketing",
    "Customer Success",
    "Customer Support",
    "Pricing",
    "Revenue Operations (RevOps)",
    "Product Management",
}

# BU normalization → canonical L1 where possible
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
    for c in _CANONICAL_BUCKETS:
        if s == c.lower():
            return c
    return _BU_SYNONYMS.get(s)

# Leadership policy guard
_LEADISH_RE = re.compile(
    r"\b(vice[-\s]?president|vp|svp|evp|president|head(?:\s+of)?|director|sr\.?\s*director)\b",
    re.I,
)
def _violates_leadership_policy(title: str, l1: Optional[str], l2: Optional[str]) -> bool:
    if not title:
        return False
    t = title.lower()
    tagged = (str(l1 or "").lower() == "leadership") or (str(l2 or "").lower() == "leadership")
    if not tagged:
        return False
    has_manager = ("manager" in t) or ("mgr" in t)
    has_leadish = _LEADISH_RE.search(t) is not None
    return (not has_leadish) or has_manager

# --- RevOps strategy nudge (keeps GPT in control) ---
_REVOPS_STRATEGY_TERMS = {
    "gtm", "go-to-market", "go to market", "strategy", "strategic",
    "revenue operations", "rev ops", "revops", "sales operations", "marketing operations",
    "enablement", "planning", "operations", "ops", "analytics", "insights", "forecast",
    "pipeline", "territory", "compensation", "crm", "data ops", "data operations", "data"
}
def _needs_revops_strategy_hint(title: str) -> bool:
    t = (title or "").lower()
    return any(k in t for k in _REVOPS_STRATEGY_TERMS)

# ---------- keyword rules from Final Titles (for L2 only) ----------
_TITLE_SYNONYMS = {
    "marketing": r"(marketing|mktg|mar)",
    "mgr": r"(mgr|manager|management|mgmt)",
    "management": r"(manager|management|mgmt|mgr)",
    "mgmt": r"(manager|management|mgmt|mgr)",
    "demand": r"(demand)",
    "gen": r"(gen|generation)",
    "campaign": r"(campaign|campaigns)",
    "program": r"(program|programme|pgm)",
    "portfolio": r"(portfolio)",
    "growth": r"(growth)",
    "brand": r"(brand)",
    "content": r"(content)",
    "sales": r"(sales)",
    "rep": r"(rep|representative|executive)",
    "account": r"(account|accounts|acct)",
    "executive": r"(executive|exec)",
    "ae": r"(ae)",
    "customer": r"(customer|client)",
    "success": r"(success)",
    "support": r"(support|help\s?desk|ticket|escalation|troubleshoot|tier[ -]?[1-4])",
    "renewal": r"(renewal|renewals|retention)",
    "farmer": r"(farmer|account management)",
    "product": r"(product|pm|product\s+(manager|owner|management|marketing))",
    "revops": r"(rev(\.|enue)?\s*ops|revops|gtm ops|sales ops|crm|sfdc|enablement|funnel|sales analyst)",
}

_GENERIC_BY_L1 = {
    "Marketing": {"marketing", "mktg", "manager", "mgr", "management", "mgmt",
                  "global", "sr", "senior", "lead", "leader", "director", "head"},
    "Customer Success": {"manager", "mgr", "management", "mgmt"},
    "Customer Support": {"manager", "mgr", "management", "mgmt"},
    "Revenue Operations (RevOps)": {"manager", "mgr", "management", "mgmt"},
    "Product Management": {"manager", "mgr", "management", "mgmt"},
    "Pricing": {"manager", "mgr", "management", "mgmt"},
}
_STOPWORDS = {"and", "&", "of", "the", "global", "intl", "sr", "senior", "jr", "associate", "assistant"}

def _phrase_to_regex_l1(phrase: str, l1: str) -> str:
    tokens = re.findall(r"[a-zA-Z]+", str(phrase).lower())
    ignore = _STOPWORDS | _GENERIC_BY_L1.get(l1, set())
    parts = []
    for tok in tokens:
        if tok in ignore:
            continue
        alt = _TITLE_SYNONYMS.get(tok)
        parts.append(fr"(?=.*\b{alt}\b)" if alt else fr"(?=.*\b{re.escape(tok)}\b)")
    return "".join(parts) + ".*" if parts else r"$^"

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

# ---------- tokenization for generic keyword scoring ----------
_CANON_MAP = {"mktg": "marketing", "mar": "marketing", "acct": "account", "pm": "product", "ops": "operations"}

def _tokenize(s: str) -> List[str]:
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9\s]+", " ", s)
    toks = [_CANON_MAP.get(w, w) for w in s.split() if w]
    bigrams = [" ".join(pair) for pair in zip(toks, toks[1:])]
    return toks + bigrams

# ---------------- taxonomy loaders ----------------
def load_taxonomy_from_frames(
    taxonomy_df: pd.DataFrame,
    rules_df: Optional[pd.DataFrame] = None,
    exact_map_df: Optional[pd.DataFrame] = None,  # unused
    *,
    examples_col: str = "Final Titles",
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
                    priority=int(row["priority"]) if "priority" in r.columns and str(row["priority"]).strip() else 100,
                    notes=(str(row["notes"]).strip() or None) if "notes" in r.columns else None,
                )
            )

    if examples_col in taxonomy_df.columns:
        for _, row in taxonomy_df.iterrows():
            l1 = str(row["L1"]).strip()
            l2 = str(row["L2"]).strip()
            examples = _split_examples(row[examples_col])
            for ex in examples:
                pat = _phrase_to_regex_l1(ex, l1)
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

# --------- keyword index from examples (from old Taxonomy_mapping) ---------
def keywords_by_l1_from_examples(
    taxonomy_df: pd.DataFrame,
    *,
    examples_col: str = "Final Titles",
) -> dict[str, dict[str, List[str]]]:
    out: dict[str, dict[str, List[str]]] = {}
    if examples_col not in taxonomy_df.columns:
        return out
    for _, r in taxonomy_df.fillna("").iterrows():
        l1 = str(r["L1"]).strip()
        l2 = str(r["L2"]).strip()
        if not (l1 and l2):
            continue
        examples = _split_examples(r[examples_col])
        tokset: set[str] = set()
        for ex in examples:
            tokset.update(_tokenize(ex))
        if tokset:
            out.setdefault(l1, {}).setdefault(l2, []).extend(sorted(tokset))
    for l1 in out:
        for l2 in out[l1]:
            out[l1][l2] = sorted(set(out[l1][l2]))
    return out

# --------- descriptions from taxonomy (from old Taxonomy_mapping) ---------
def extract_descriptions_from_taxonomy_df(
    taxonomy_df: pd.DataFrame,
    *,
    examples_col: str = "Final Titles",
    per_l1_max: int = 6,
    per_l2_max_examples: int = 3,
    l1_clip_chars: int = 600,
) -> tuple[dict[str, str], dict[str, str]]:
    l2_desc: dict[str, str] = {}
    l1_desc: dict[str, str] = {}

    df = taxonomy_df.fillna("")
    use_examples = examples_col in df.columns
    use_desc = ("Description" in df.columns) if not use_examples else False
    if not use_examples and not use_desc:
        return l1_desc, l2_desc

    if use_examples:
        for l1, group in df.groupby("L1", sort=False):
            bullets: List[str] = []
            for _, r in group.iterrows():
                l2 = str(r["L2"]).strip()
                ex_list = _split_examples(r[examples_col])[:per_l2_max_examples]
                if l2 and ex_list:
                    l2_desc.setdefault(l2, "Examples: " + "; ".join(ex_list))
                    if len(bullets) < per_l1_max:
                        bullets.append(f"{l2} – " + " | ".join(ex_list))
            if bullets:
                text = "; ".join(bullets)
                if len(text) > l1_clip_chars:
                    text = text[: l1_clip_chars - 1] + "…"
                l1_desc[str(l1).strip()] = text
    else:
        seen_l2 = set()
        for l1, group in df.groupby("L1", sort=False):
            bullets: List[str] = []
            for _, r in group.iterrows():
                l2 = str(r["L2"]).strip()
                d  = str(r["Description"]).strip()
                if not (l2 and d):
                    continue
                norm_l2 = l2.lower()
                if norm_l2 not in seen_l2:
                    l2_desc.setdefault(l2, d)
                    seen_l2.add(norm_l2)
                if len(bullets) < per_l1_max:
                    bullets.append(f"{l2} – {d}")
            if bullets:
                text = "; ".join(bullets)
                if len(text) > l1_clip_chars:
                    text = text[: l1_clip_chars - 1] + "…"
                l1_desc[str(l1).strip()] = text

    return l1_desc, l2_desc

# ------------- anchor-based disambiguation (commercial buckets) -------------
ANCHORS = {
    "Sales": {
        "must": r"\b(sales|account\s+executive|\bae\b|bdr|sdr|business\s+development|rep|representative)\b",
        "contra": r"\b(renewal|renewals|retention|account\s+management|customer\s+success|csm|success\s+manager)\b",
        "contra_to": "Customer Success",
        "fallbacks": [
            ("Customer Success", r"\b(renewal|renewals|retention|account\s+management|customer\s+success|csm)\b"),
        ],
    },
    "Product Management": {
        "must":  r"\b(product(s)?|product\s+(manager|owner|management|marketing)|\bpm\b)\b",
        "fallbacks": [("Marketing", r"\b(marketing|campaign|demand|brand|portfolio)\b")],
    },
    "Marketing": {
        "must":  r"\b(marketing|campaign|demand|brand|content|growth)\b",
        "fallbacks": [("Product Management", r"\b(product\s+(marketing|manager|owner)|\bpm\b)\b")],
    },
    "Customer Support": {
        "must":  r"\b(support|help ?desk|ticket|escalation|troubleshoot|tier[ -]?[1-4])\b",
        "fallbacks": [("Customer Success", r"\b(customer success|csm|onboarding|adoption|health|renewal(s)?)\b")],
    },
    "Customer Success": {
        "must":  r"\b(customer success|csm|onboarding|adoption|health|renewal(s)?)\b",
        "fallbacks": [("Customer Support", r"\b(support|help ?desk|ticket|escalation|troubleshoot|tier[ -]?[1-4])\b")],
    },
    "Revenue Operations (RevOps)": {
        "must":  r"\b(rev(\.|enue)?\s*ops|revops|gtm ops|sales ops|crm|sfdc|enablement|funnel|operations|ops)\b",
        "fallbacks": [("Pricing", r"\b(pricing|price ops|deal desk)\b"), ("Sales", r"\b(pipeline|quota|forecast)\b")],
    },
    "Pricing": {
        "must": r"\b(pricing|price\s*strategy|deal\s*desk|cpq|configure[-\s]*price[-\s]*quote|"
                r"quote[-\s]*to[-\s]*cash|q2c|discount(?:ing)?|margin|profitabilit(?:y|ies)|"
                r"rate\s*card|pricebook|list\s*price|pricing\s*ops?|price\s*ops|deal\s*pricing)\b",
        "fallbacks": [
            ("Revenue Operations (RevOps)", r"\b(rev(?:\.|enue)?\s*ops|revops|gtm\s*ops|sales\s*ops|"
                                            r"crm|sfdc|enablement|operations|ops)\b"),
            ("Sales", r"\b(contracts?|quote|quoting|proposal|negotiat(?:e|ion|ions)|opportunit(?:y|ies))\b"),
        ],
    },
}

def _apply_disambig(l1: str, title: str, bu_norm: Optional[str]) -> tuple[str, Optional[str], float]:
    cfg = ANCHORS.get(l1)
    if not cfg:
        return l1, None, 1.0
    t = (title or "").lower()
    contra_pat = cfg.get("contra")
    if contra_pat and re.search(contra_pat, t, re.I):
        to = cfg.get("contra_to")
        if to:
            return to, f"{l1}→{to}-neg", 0.92
    must_pat = cfg.get("must")
    if must_pat and re.search(must_pat, t, re.I):
        return l1, None, 1.0
    for fb_l1, fb_pat in cfg.get("fallbacks", []):
        if bu_norm == fb_l1 or (fb_pat and re.search(fb_pat, t, re.I)):
            return fb_l1, f"{l1}→{fb_l1}-disambig", 0.92
    return l1, None, 1.0

# -------- Non-commercial anchors (guided by BLS SOC major groups & O*NET) --------
_NC_LABELS: List[str] = [
    "Human Resources", "Finance / Accounting", "Legal / Compliance",
    "Information Technology", "Engineering / R&D", "Manufacturing / Production",
    "Supply Chain / Procurement", "Logistics / Warehousing / Distribution",
    "Facilities / Real Estate / Maintenance", "Construction",
    "EHS / Safety / Security", "Quality", "Admin / Office / Clerical",
    "Education / Training (L&D)", "PR / Communications", "Healthcare",
    "Farming / Forestry", "Transportation", "Executive / General Management",
    "Other (NC)"
]

_COMMERCIAL_GUARD_RE = re.compile(
    r"\b(sales|account\s+executive|bdr|sdr|business\s+development|marketing|"
    r"customer\s+success|customer\s+support|rev(\.|enue)?\s*ops|rev\s*ops|"
    r"product\s+management|pricing)\b",
    re.I
)

_NC_ANCHOR_PATTERNS = {
    "Human Resources": r"\b(human\s*resources?|hr|people\s*ops|talent\s*(acquisition|management)|recruit(ing|ment)|payroll|benefits|comp(ensation)?)\b",
    "Finance / Accounting": r"\b(finance|financial|account(ing|ant)|accounts?\s*payable|accounts?\s*receivable|treasury|controller|audit|auditor|tax|fp&?a)\b",
    "Legal / Compliance": r"\b(legal|attorney|lawyer|counsel|paralegal|compliance|regulatory)\b",
    "Information Technology": r"\b(it|information\s*technology|systems?\s*admin|sysadmin|network|help\s*desk|desktop\s*support|infrastructure|devops|sre|database|dba|cyber|infosec|security\s*(analyst|engineer))\b",
    "Engineering / R&D": r"\b((?<!sales\s)engineer(ing)?|developer|software|hardware|firmware|scientist|research|r&d|lab|chemist|physicist|biologist|data\s*scientist|qa|quality\s*assurance)\b",
    "Manufacturing / Production": r"\b(manufactur(?:e|ing)|production|assembly|fabrication|machinist|operator|plant|factory|c\.?n\.?c|cnc|industrial)\b",
    "Supply Chain / Procurement": r"\b(supply\s*chain|procure(?:ment|r)|purchasing|buyer|sourcing|category\s*manager|demand\s*plan|material\s+plan)\b",
    "Logistics / Warehousing / Distribution": r"\b(logistics|warehouse|warehousing|inventory|fulfillment|shipping|receiving|distribution|dispatcher|transport|fleet|routing)\b",
    "Facilities / Real Estate / Maintenance": r"\b(facilit(?:y|ies)|real\s*estate|property\s*manage(?:r|ment)|building|grounds|custodian|janitor|maintenance|mechanic|hvac|electrician|plumber|carpenter)\b",
    "Construction": r"\b(construction|foreman|estimator|site\s*manager|quantity\s*surveyor|qs|civil\s*engineer)\b",
    "EHS / Safety / Security": r"\b(ehs|hse|safety|environment(al)?\s*health|oh&s|osha|security|guard|loss\s*prevention)\b",
    "Quality": r"\b(quality\s*(assurance|control)|qa|qc|six\s*sigma|lean)\b",
    "Admin / Office / Clerical": r"\b(administrative|admin|office|receptionist|cleri?cal|\bclerk\b|secretary|assistant|data\s*entry|mailroom)\b",
    "Education / Training (L&D)": r"\b(learning\s*&\s*development|l&d|learning\s+and\s+development|trainer|training|instructor|curriculum|academy|university|teacher|coach)\b",
    "PR / Communications": r"\b(public\s*relations|(?:^|\s)pr\s|communications?\b|comms\b|media\s*relations|press|spokes(?:person|man|woman))\b",
    "Healthcare": r"\b(nurse|physician|clinical|medical|healthcare|dental|pharmac(y|ist)|clinic|hospital)\b",
    "Farming / Forestry": r"\b(agricultur(?:e|al)|farm(?:er|hand)?|grower|forestry|logger)\b",
    "Transportation": r"\b(driver|truck\s*driver|courier|dispatcher|pilot|captain|sailor|maritime)\b",
    "Executive / General Management": r"\b(ceo|chief\s+\w+|coo|cfo|cio|cto|cso|chro|president|executive|board|chair|founder|general\s*manager|gm|managing\s*director)\b",
    "Other (NC)": r".*",
}
_NC_ANCHOR_RES = {k: re.compile(v, re.I) for k, v in _NC_ANCHOR_PATTERNS.items()}

_NC_HINT_TERMS: Dict[str, List[str]] = {
    "Human Resources": ["HR", "people ops", "recruiting", "talent", "payroll", "benefits"],
    "Finance / Accounting": ["AP/AR", "controller", "treasury", "audit", "tax", "FP&A"],
    "Legal / Compliance": ["legal", "counsel", "paralegal", "compliance", "regulatory"],
    "Information Technology": ["IT", "sysadmin", "network", "help desk", "infrastructure", "DevOps", "SRE", "security"],
    "Engineering / R&D": ["engineer", "developer", "software", "hardware", "scientist", "R&D", "lab"],
    "Manufacturing / Production": ["manufacturing", "production", "assembly", "machinist", "CNC", "plant"],
    "Supply Chain / Procurement": ["supply chain", "procurement", "purchasing", "buyer", "sourcing", "planning"],
    "Logistics / Warehousing / Distribution": ["logistics", "warehouse", "inventory", "shipping", "distribution"],
    "Facilities / Real Estate / Maintenance": ["facilities", "real estate", "property", "maintenance", "HVAC", "electrician"],
    "Construction": ["construction", "foreman", "estimator", "site manager", "quantity surveyor"],
    "EHS / Safety / Security": ["EHS", "HSE", "safety", "OSHA", "security", "loss prevention"],
    "Quality": ["quality assurance", "quality control", "QA", "QC", "Six Sigma", "Lean"],
    "Admin / Office / Clerical": ["administrative", "office", "receptionist", "clerical", "clerk", "secretary"],
    "Education / Training (L&D)": ["learning", "development", "trainer", "training", "instructor", "curriculum"],
    "PR / Communications": ["public relations", "PR", "communications", "media", "press"],
    "Healthcare": ["nurse", "physician", "clinical", "medical", "hospital"],
    "Farming / Forestry": ["agriculture", "farmer", "grower", "forestry"],
    "Transportation": ["driver", "courier", "pilot", "maritime"],
    "Executive / General Management": ["CEO", "COO", "CFO", "GM", "Managing Director"],
    "Other (NC)": [],
}

def _nc_candidates(title: str, bu_raw: Optional[str]) -> List[str]:
    text = f"{(bu_raw or '')} | {(title or '')}".lower()
    out: List[str] = []
    for label, rx in _NC_ANCHOR_RES.items():
        if label == "Executive / General Management":
            if rx.search(text) and not _COMMERCIAL_GUARD_RE.search(text):
                out.append(label)
        else:
            if rx.search(text):
                out.append(label)
    return list(dict.fromkeys(out))

# ---------------- keyword scorer for L2 (generic, taxonomy-first) ----------------
def _score_l2_by_keywords(
    title: str,
    l1_final: str,
    allowed_l2: List[str],
    kw_by_l1: Optional[dict[str, dict[str, List[str]]]],
    *,
    require_decisive: bool = True,
) -> tuple[Optional[str], float]:
    if not kw_by_l1:
        return None, 0.0
    kw_map = kw_by_l1.get(l1_final) or {}
    if not kw_map:
        return None, 0.0

    from collections import Counter
    df = Counter()
    for l2 in allowed_l2:
        toks = set(kw_map.get(l2, []))
        for t in toks:
            df[t] += 1
    N = max(1, len(allowed_l2))

    title_toks = set(_tokenize(title))

    scores: Dict[str, float] = {}
    for l2 in allowed_l2:
        toks = set(kw_map.get(l2, []))
        overlap = title_toks & toks
        score = 0.0
        for t in overlap:
            idf = math.log((N + 1) / (1 + df.get(t, 0))) + 1.0
            score += idf
        scores[l2] = score

    best_l2, best = max(scores.items(), key=lambda kv: kv[1]) if scores else (None, 0.0)
    second = sorted(scores.values(), reverse=True)[1] if len(scores) > 1 else 0.0

    if require_decisive:
        if best >= 1.0 and (best - second) >= 0.3:
            return best_l2, best
        return None, 0.0
    else:
        return (best_l2, best) if best > 0 else (None, 0.0)

def _extract_json_block(text: str):
    if not text:
        raise ValueError("empty model reply")
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.I | re.M)
    m = re.search(r"(\[.*\]|\{.*\})", text, flags=re.S)
    if not m:
        raise ValueError("no JSON found in model reply")
    return json.loads(m.group(1))

def _chunk_iter(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i+n]

def _print_progress(processed: int, total: int, *, label: str = "[progress]") -> None:
    if total <= 0:
        return
    pct = 100.0 * processed / total
    print(f"\r{label} {processed}/{total} ({pct:.1f}%)", end="", flush=True)

# ================================= MAIN =================================
def batch_map_employees(
    employees_df: "pd.DataFrame",
    title_col: str,
    bu_col: Optional[str],
    taxonomy: Dict[str, List[str]],
    rules: List["Rule"],
    exact: Dict[Tuple[str, Optional[str]], Tuple[str, str]] | None,  # unused
    *,
    min_confidence: float = 0.85,
    kw_by_l1: Optional[dict[str, dict[str, List[str]]]] = None,
    batch_size: int = 16,
    model: str = "gpt-4o",
    temperature: float = 0.0,
    progress: bool = True,
    progress_label: str = "[progress]",
    generate_rationale: bool = False,
    rationale_model: Optional[str] = None,
    use_nc_l2_gpt: bool = False,
) -> "pd.DataFrame":

    if title_col not in employees_df.columns:
        raise KeyError(f"employees_df missing title column: {title_col}")
    has_bu = bool(bu_col) and (bu_col in employees_df.columns)

    # -------- L1 weighting regime --------
    if has_bu:
        present = {_normalize_bucket(v) for v in employees_df[bu_col].dropna().astype(str)}
        present = {p for p in present if p in _CANONICAL_BUCKETS}
        n_buckets = len(present)
    else:
        n_buckets = 0
    if n_buckets <= 4:
        w_title, w_bu = 0.50, 0.50
    elif n_buckets <= 6:
        w_title, w_bu = 0.30, 0.70
    else:
        w_title, w_bu = 0.0, 1.0  # lock to BU if canonical

    # -------- Prep outputs --------
    out = employees_df.copy()
    for col in ("Mapped_L1","Mapped_L2","MapSource","Rationale","Reclassified_L1","Reclassified_L2"):
        out[col] = None
    out["Confidence"] = 0.0

    # -------- Pre-index rules by L1 --------
    rules_by_l1: Dict[str, List["Rule"]] = {}
    for r in (rules or []):
        rules_by_l1.setdefault(r.l1, []).append(r)
    for l1 in list(rules_by_l1.keys()):
        rules_by_l1[l1].sort(key=lambda r: (r.priority, len(r.pattern)), reverse=True)

    client = OpenAI()
    total_rows = len(out)

    # ================================= L1 (batched) =================================
    l1_items = []
    for idx, row in out.iterrows():
        title = str(row[title_col]) if pd.notna(row[title_col]) else ""
        bu_raw = str(row[bu_col]) if has_bu and pd.notna(row[bu_col]) else ""
        bu_norm = _normalize_bucket(bu_raw)
        l1_items.append({
            "id": int(idx),
            "title": title,
            "bu_raw": bu_raw,
            "bu_norm": bu_norm or "",
            "text": f"{bu_raw.strip()} | {title}".strip(" |"),
            "lock_bu": (w_bu == 1.0 and (bu_norm in _CANONICAL_BUCKETS)),
            "nc_candidates": _nc_candidates(title, bu_raw),
        })

    l1_results: Dict[int, Tuple[str, float, str]] = {}

    done = 0
    for chunk in _chunk_iter(l1_items, batch_size):
        locked = [it for it in chunk if it["lock_bu"]]
        to_model = [it for it in chunk if not it["lock_bu"]]

        for it in locked:
            l1_results[it["id"]] = (it["bu_norm"], 0.99, "ai-L1(bu-locked)")

        if to_model:
            payload = []
            for it in to_model:
                payload.append({
                    "id": it["id"],
                    "text": it["text"],
                    "hint_revops": bool(_needs_revops_strategy_hint(it["title"])),
                    "nc_candidates": it["nc_candidates"],
                })

            l1_choices = sorted(_CANONICAL_BUCKETS | {"Non-commercial", "Other"})
            nc_hints = {lab: _NC_HINT_TERMS.get(lab, []) for lab in _NC_LABELS}

            prompt = f"""
Map each item to ONE L1 bucket strictly from this list:
{l1_choices}

Signals:
- BEFORE the pipe is the job family group (BU)
- AFTER the pipe is the job title
Combine evidence with these weights: title={w_title:.2f}, bu={w_bu:.2f}.
If weights are equal, let the title evidence break ties. If BU contradicts the title, prefer the title.

Guidance:
- The taxonomy provides L2 rules for COMMERCIAL GTM buckets only ({sorted(list(_CANONICAL_BUCKETS))}).
- You MAY choose "Non-commercial" if the role is clearly an internal/support function.
- You'll receive "nc_candidates" derived from labor-standard anchors; they are hints only.
- If the text contains explicit GTM terms (sales, marketing, customer success/support, pricing, rev ops, product management),
  prefer a COMMERCIAL bucket over Non-commercial.

Return ONLY JSON with objects: {{"id": <id>, "l1": "<one of the choices>", "confidence": 0-1}}.

NC hints (keywords per NC family):
{json.dumps(nc_hints, ensure_ascii=False)}

Items:
{json.dumps(payload, ensure_ascii=False, indent=2)}
""".strip()

            resp = client.chat.completions.create(
                model=model,
                temperature=temperature,
                messages=[{"role":"user","content":prompt}],
            )
            data = _extract_json_block(resp.choices[0].message.content or "")
            norm_keys = {k.lower(): k for k in taxonomy.keys()}
            for obj in data:
                i  = int(obj.get("id"))
                l1 = str(obj.get("l1") or "").strip()
                conf = float(obj.get("confidence") or 0.0)
                if l1.lower() in norm_keys:
                    l1 = norm_keys[l1.lower()]
                l1_results[i] = (l1, conf, "ai-L1(combined)")

        done += len(chunk)
        if progress:
            _print_progress(done, total_rows, label="[L1]")

    if progress: print()

    # Leadership policy + anchor disambiguation (commercial only)
    for it in l1_items:
        idx, title_raw = it["id"], it["title"]
        bu_norm = it["bu_norm"] or None
        l1_final, l1_conf, l1_src = l1_results[idx]

        if l1_final != "Non-commercial":
            if _violates_leadership_policy(title_raw, l1_final, None):
                l1_final = "Other"
                l1_src   = l1_src + "(policy)"
                l1_conf  = min(l1_conf, min_confidence - 1e-6)
            if l1_final == "Other" and l1_conf >= min_confidence:
                l1_conf = min(l1_conf, min_confidence - 1e-6)

            l1_new, tag, cap = _apply_disambig(l1_final, title_raw, bu_norm)
            if tag and l1_new != l1_final:
                l1_final, l1_src = l1_new, f"{l1_src}+{tag}"
                l1_conf = min(l1_conf, cap)

        out.at[idx, "Mapped_L1"]    = l1_final
        out.at[idx, "_l1_conf_tmp"] = float(l1_conf)
        out.at[idx, "_l1_src_tmp"]  = l1_src

    # ================================= L2 =================================
    def _try_rules(l1: str, title: str, bu: Optional[str]) -> Optional[Tuple[str,str]]:
        for r in rules_by_l1.get(l1, []):
            if _match_rule(title, r, bu):
                return r.l2, "rule-l2"
        return None

    need_l2_ai: List[dict] = []
    need_nc_l2_ai: List[dict] = []

    for idx, row in out.iterrows():
        title_raw = str(row[title_col]) if pd.notna(row[title_col]) else ""
        bu_raw    = str(row[bu_col]) if has_bu and pd.notna(row[bu_col]) else None
        l1_final  = row["Mapped_L1"]

        if l1_final == "Non-commercial":
            nc_cands = _nc_candidates(title_raw, bu_raw)
            if use_nc_l2_gpt:
                need_nc_l2_ai.append({
                    "id": int(idx),
                    "text": f"{(bu_raw or '').strip()} | {title_raw}".strip(" |"),
                    "nc_candidates": nc_cands,
                    "allowed_l2": [lab for lab in _NC_LABELS],
                })
                out.at[idx, "_l2_tmp"] = ""
                out.at[idx, "_l2_src_tmp"] = ""
                out.at[idx, "_l2_conf_tmp"] = 0.7
            else:
                chosen = nc_cands[0] if nc_cands else "Other (NC)"
                out.at[idx, "_l2_tmp"] = chosen
                out.at[idx, "_l2_src_tmp"] = "nc-anchors(rule)"
                out.at[idx, "_l2_conf_tmp"] = 0.85
            continue

        allowed_l2 = taxonomy.get(l1_final, [])
        l2_final, l2_src, l2_conf = "", "", 0.6

        if l1_final and allowed_l2:
            hit = _try_rules(l1_final, title_raw, bu_raw)
            if hit:
                l2_final, l2_src, l2_conf = hit[0], hit[1], 0.90
            else:
                best_dec, score_dec = _score_l2_by_keywords(
                    title_raw, l1_final, allowed_l2, kw_by_l1, require_decisive=True
                )
                if best_dec:
                    l2_final, l2_src = best_dec, "kw-l2"
                    l2_conf = min(0.95, 0.80 + min(0.15, score_dec/4.0))
                else:
                    best_any, _ = _score_l2_by_keywords(
                        title_raw, l1_final, allowed_l2, kw_by_l1, require_decisive=False
                    )
                    need_l2_ai.append({
                        "id": int(idx),
                        "text": f"{(bu_raw or '').strip()} | {title_raw}".strip(" |"),
                        "l1": l1_final,
                        "allowed_l2": allowed_l2,
                        "kw_best_any": best_any,
                    })

        out.at[idx, "_l2_tmp"] = l2_final
        out.at[idx, "_l2_src_tmp"] = l2_src
        out.at[idx, "_l2_conf_tmp"] = float(l2_conf)

    # GPT for remaining COMMERCIAL L2
    total_ai = len(need_l2_ai); done_ai = 0
    for chunk in _chunk_iter(need_l2_ai, batch_size):
        payload = []
        for it in chunk:
            kw_map = (kw_by_l1 or {}).get(it["l1"], {})
            lines = [f"{l2}: {', '.join(kw_map.get(l2, [])[:8])}" for l2 in it["allowed_l2"] if kw_map.get(l2)]
            payload.append({
                "id": it["id"], "text": it["text"], "l1": it["l1"],
                "allowed_l2": it["allowed_l2"], "kw_hint": " | ".join(lines) if lines else ""
            })

        prompt = """
STAGE=L2 (Commercial). For each item, choose EXACTLY ONE L2 strictly from allowed_l2.
Use the title keywords and the provided hints. Do NOT hallucinate or invent labels.
Return only JSON: [{"id": <id>, "l2": "<L2>", "confidence": 0-1}, ...].

Items:
""".strip() + "\n" + json.dumps(payload, ensure_ascii=False, indent=2)

        resp = client.chat.completions.create(
            model=model, temperature=temperature,
            messages=[{"role":"user","content":prompt}],
        )
        data = _extract_json_block(resp.choices[0].message.content or "")
        by_id = {int(x["id"]): x for x in data if "id" in x}

        for it in chunk:
            idx = it["id"]
            res = by_id.get(idx, {})
            l2  = str(res.get("l2") or "").strip()
            conf = float(res.get("confidence") or 0.6)
            allowed = it["allowed_l2"]

            if not l2 or l2 not in allowed:
                l2 = it["kw_best_any"] or (allowed[0] if allowed else "")
                conf = max(conf, 0.88)
                src = "kw-l2(override-other)"
            else:
                src = "ai-L2"

            out.at[idx, "_l2_tmp"] = l2
            out.at[idx, "_l2_src_tmp"] = src
            out.at[idx, "_l2_conf_tmp"] = float(conf)

        done_ai += len(chunk)
        if progress:
            _print_progress(done_ai, total_ai or 1, label="[L2-ai]")

    if progress: print()

    # GPT for NON-COMMERCIAL L2
    total_nc = len(need_nc_l2_ai); done_nc = 0
    for chunk in _chunk_iter(need_nc_l2_ai, batch_size):
        payload = []
        for it in chunk:
            cand = it["nc_candidates"] or _NC_LABELS
            hints = {lab: _NC_HINT_TERMS.get(lab, []) for lab in cand}
            payload.append({
                "id": it["id"],
                "text": it["text"],
                "allowed_l2": _NC_LABELS,
                "nc_candidates": it["nc_candidates"],
                "hints": hints,
            })

        prompt_nc = """
STAGE=L2 (Non-commercial). For each item where L1="Non-commercial", choose EXACTLY ONE L2
strictly from allowed_l2 (non-commercial families). Use the title/BU plus the provided anchor hints.
Prefer a label from nc_candidates when it fits; otherwise choose the best overall.
Do NOT invent labels. Return only JSON:
[{"id": <id>, "l2": "<one of allowed_l2>", "confidence": 0-1}, ...].

Items:
""".strip() + "\n" + json.dumps(payload, ensure_ascii=False, indent=2)

        resp = client.chat.completions.create(
            model=model, temperature=temperature,
            messages=[{"role":"user","content":prompt_nc}],
        )
        data = _extract_json_block(resp.choices[0].message.content or "")
        by_id = {int(x["id"]): x for x in data if "id" in x}

        for it in chunk:
            idx = it["id"]
            res = by_id.get(idx, {})
            l2  = str(res.get("l2") or "").strip()
            conf = float(res.get("confidence") or 0.7)
            allowed = it["allowed_l2"]
            if not l2 or l2 not in allowed:
                cand = (it["nc_candidates"][0] if it["nc_candidates"] else "Other (NC)")
                l2 = cand if cand in allowed else "Other (NC)"
                conf = max(conf, 0.85)
                src = "nc-anchors(fallback)"
            else:
                src = "ai-NC-L2"

            out.at[idx, "_l2_tmp"] = l2
            out.at[idx, "_l2_src_tmp"] = src
            out.at[idx, "_l2_conf_tmp"] = float(conf)

        done_nc += len(chunk)
        if progress:
            _print_progress(done_nc, total_nc or 1, label="[L2-nc-ai]")

    if progress: print()

    # ================================= Finalisation =================================
    records: List[dict] = []
    for idx, row in out.iterrows():
        title_raw = str(row[title_col]) if pd.notna(row[title_col]) else ""
        bu_raw    = str(row[bu_col]) if has_bu and pd.notna(row[bu_col]) else None

        l1_final = str(row["Mapped_L1"] or "")
        l1_conf  = float(row["_l1_conf_tmp"] or 0.0)
        l1_src   = str(row["_l1_src_tmp"] or "")

        allowed_l2 = taxonomy.get(l1_final, [])
        l2_final = str(row["_l2_tmp"] or "")
        l2_conf  = float(row["_l2_conf_tmp"] or 0.6)
        l2_src   = str(row["_l2_src_tmp"] or "")

        if l1_final != "Non-commercial" and _violates_leadership_policy(title_raw, l1_final, l2_final):
            nonlead = [x for x in allowed_l2 if x.lower() != "leadership"]
            best_nonlead, _ = _score_l2_by_keywords(
                title_raw, l1_final, nonlead or allowed_l2, kw_by_l1, require_decisive=False
            )
            if best_nonlead:
                l2_final = best_nonlead
            elif nonlead:
                l2_final = nonlead[0]
            elif allowed_l2:
                l2_final = allowed_l2[0]
            l2_conf = min(l2_conf, min_confidence - 1e-6)

        overall_conf = round((float(l1_conf or 0) + float(l2_conf or 0)) / 2, 3)
        if l1_final == "Other" and overall_conf >= min_confidence:
            overall_conf = round(min_confidence - 1e-6, 3)

        if l1_final == "Non-commercial":
            out_l1, out_l2 = "Non-commercial", (l2_final or "Other (NC)")
        elif l1_final not in taxonomy:
            out_l1, out_l2 = "Other", ""
        else:
            out_l1, out_l2 = (l1_final, l2_final) if l2_final in allowed_l2 else (l1_final, "")

        source = f"{l1_src}+{l2_src}" if l2_src else l1_src

        re_l1 = re_l2 = None
        if overall_conf < min_confidence and l1_final != "Non-commercial":
            re_l1 = bu_raw if (bu_raw and str(bu_raw).strip()) else "Other"
            re_l2 = "Other"
            source = source + "(low-conf->reclass)"

        records.append({
            **{c: row[c] for c in employees_df.columns},
            "Mapped_L1": out_l1,
            "Mapped_L2": out_l2,
            "MapSource": source,
            "Confidence": overall_conf,
            "Rationale": None,
            "Reclassified_L1": re_l1,
            "Reclassified_L2": re_l2,
        })

    result = pd.DataFrame.from_records(records)

    if generate_rationale:
        r_model = rationale_model or model
        need_rat = []
        for i, r in result.iterrows():
            if r["Mapped_L1"] and r["Mapped_L2"]:
                need_rat.append({
                    "id": int(i),
                    "title": str(r[title_col]),
                    "bu": str(r[bu_col]) if has_bu and pd.notna(r[bu_col]) else "",
                    "l1": r["Mapped_L1"],
                    "l2": r["Mapped_L2"],
                })

        done_r = 0
        for chunk in _chunk_iter(need_rat, batch_size):
            prompt = """Write a concise 1–2 sentence rationale for each item explaining why the title fits the mapped L1/L2. 
Avoid mentioning AI, models, weights, or confidence. Return ONLY JSON:
[{"id": <id>, "rationale": "<text>"} ...].

Items:
""" + json.dumps(chunk, ensure_ascii=False, indent=2)

            resp = client.chat.completions.create(
                model=r_model, temperature=0.0,
                messages=[{"role":"user","content":prompt}],
            )
            data = _extract_json_block(resp.choices[0].message.content or "")
            by_id = {int(x["id"]): x.get("rationale") for x in data if "id" in x}
            for it in chunk:
                rid = it["id"]
                rat = by_id.get(rid)
                if rat:
                    result.at[rid, "Rationale"] = rat

            done_r += len(chunk)
            if progress:
                _print_progress(done_r, len(need_rat) or 1, label="[rationale]")

        if progress: print()

    if progress:
        _print_progress(len(result), len(result), label=progress_label)
        print("\n[batch] finalised")

    return result
