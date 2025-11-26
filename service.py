#!/usr/bin/env python3
"""
Shared services & classes:
- TaggerConfig  (dataclass)
- Rule          (dataclass for deterministic rules)
- RoleTagger    (OpenAI-based classifier using Structured Outputs)
- INSTRUCTIONS_TEMPLATE + build_instructions()
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Any, Optional
import json
import time

# Optional import (so non-AI flows still work without openai installed)
try:
    from openai import OpenAI
except Exception:  # openai not installed or unavailable
    OpenAI = None  # type: ignore


# ----------------------- Config & Prompt -----------------------

INSTRUCTIONS_TEMPLATE = """
You are an expert job taxonomy classifier. Map each role to an L1 and L2 tag.

Rules
- Use ONLY the provided taxonomy for L1 and L2 choices.
- Choose the SINGLE best L1 and the SINGLE best L2 under that L1.
- If the role clearly fits multiple L2s under the same L1, pick the most
  specific and commonly accepted subcategory.
- If there's no reasonable match, return L1="Other" and L2="Other".
- Be strict: do not invent new categories or synonyms not in the taxonomy.
- Keep rationale to ≤ 1–2 sentences.

Taxonomy (JSON):
{taxonomy_json}
""".strip()


def build_instructions(
    taxonomy: dict,
    l1_desc: dict | None = None,
    l2_desc: dict | None = None,
    policy_hint: str = "",
) -> str:
    import json
    base = (
        "You classify job titles into L1/L2 from the given taxonomy and return JSON only.\n"
        + "Be conservative: if uncertain, avoid Leadership.\n\n"
        + "Taxonomy:\n"
        + json.dumps(taxonomy, ensure_ascii=False)
    )
    extra = []
    if l1_desc:
        extra.append("\nL1 descriptions:\n" + json.dumps(l1_desc, ensure_ascii=False, indent=2))
    if l2_desc:
        extra.append("\nL2 descriptions:\n" + json.dumps(l2_desc, ensure_ascii=False, indent=2))
    if policy_hint:
        extra.append("\nTAGGING POLICY:\n" + policy_hint.strip())
    return base + "".join(extra)

@dataclass
class TaggerConfig:
    model: str = "gpt-4o-mini"
    temperature: float = 0.0
    max_retries: int = 5
    retry_backoff_seconds: float = 2.0


@dataclass
class Rule:
    """Row from the Excel `rules` sheet."""
    l1: str
    l2: str
    pattern: str
    match_type: str  # equals | contains | regex
    bu: Optional[str]
    cost_center: Optional[str]
    priority: int
    notes: Optional[str]


class RoleTagger:
    """
    OpenAI Responses API classifier with Structured Outputs (JSON Schema).
    Returns: {"l1": str, "l2": str, "confidence": float, "rationale": str?}
    """

    def __init__(
        self,
        client: f"OpenAI",
        taxonomy: Dict[str, List[str]],
        cfg: TaggerConfig,
        l1_desc: Optional[Dict[str, str]] = None,
        l2_desc: Optional[Dict[str, str]] = None,
        policy_hint: str = ""
    ):
        if client is None or OpenAI is None:
            raise RuntimeError("OpenAI SDK client not available. Install `openai` and set OPENAI_API_KEY.")
        self.client = client
        self.taxonomy = taxonomy
        self.cfg = cfg
        self.schema = self._build_schema(taxonomy)
        self.instructions = build_instructions(taxonomy, l1_desc=l1_desc, l2_desc=l2_desc, policy_hint=policy_hint)

        # Precompute for validation/correction
        self.valid_l1 = set(taxonomy.keys()) | {"Other"}
        all_l2 = {l2 for l2s in taxonomy.values() for l2 in l2s}
        self.valid_l2 = set(all_l2) | {"Other"}
        self.l2_to_l1 = {}
        for l1, l2s in taxonomy.items():
            for l2 in l2s:
                self.l2_to_l1.setdefault(l2, set()).add(l1)

    @staticmethod
    # in service.py, RoleTagger._build_schema()
    def _build_schema(taxonomy: Dict[str, List[str]]) -> Dict[str, Any]:
        all_l2 = sorted({l2 for l2s in taxonomy.values() for l2 in l2s})
        props = {
            "l1": {"type": "string", "enum": sorted(list(taxonomy.keys())) + ["Other"]},
            "l2": {"type": "string", "enum": all_l2 + ["Other"]},
            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "rationale": {"type": "string"},
        }
        return {
            "type": "object",
            "properties": props,
            "required": list(props.keys()),      # <-- include ALL keys
            "additionalProperties": False,
        }

    @staticmethod
    def _format_user_text(title: str, description: Optional[str]) -> str:
        parts = [f"Title: {title.strip()}"]
        if description and description.strip():
            parts.append("Description:\n" + description.strip())
        return "\n\n".join(parts)

    def _post_validate(self, data: Dict[str, Any]) -> Dict[str, Any]:
        l1 = data.get("l1")
        l2 = data.get("l2")
        if l1 not in self.valid_l1:
            l1 = "Other"
        if l2 not in self.valid_l2:
            l2 = "Other"
        if l2 != "Other" and l1 != "Other":
            l1s_for_l2 = self.l2_to_l1.get(l2, set())
            if l1 not in l1s_for_l2 and len(l1s_for_l2) == 1:
                l1 = next(iter(l1s_for_l2))
            elif l1 not in l1s_for_l2 and len(l1s_for_l2) == 0:
                l1, l2 = "Other", "Other"
        try:
            c = float(data.get("confidence", 0.0))
            data["confidence"] = max(0.0, min(1.0, c))
        except Exception:
            data["confidence"] = 0.0
        data["l1"], data["l2"] = l1, l2
        return data

        # service.py  (inside RoleTagger.tag)
    def tag(self, title: str, description: Optional[str] = None) -> Dict[str, Any]:
        if OpenAI is None:
            raise RuntimeError("OpenAI SDK not installed. Install openai and set OPENAI_API_KEY.")
        user_text = self._format_user_text(title, description)
    
        backoff = self.cfg.retry_backoff_seconds
        last_err = None
        for attempt in range(1, self.cfg.max_retries + 1):
            try:
                common = dict(
                    model=self.cfg.model,
                    input=[
                        {"role": "system", "content": self.instructions},
                        {"role": "user", "content": user_text},
                    ],
                    temperature=self.cfg.temperature,
                )
                try:
                    # Newer SDKs
                    response = self.client.responses.create(
                        **common,
                        response_format={
                            "type": "json_schema",
                            "json_schema": {
                                "name": "role_tags",
                                "schema": self.schema,
                                "strict": True,
                            },
                        },
                    )
                except TypeError:
                    # Older SDKs
                    response = self.client.responses.create(
                        **common,
                        text={
                            "format": {
                                "type": "json_schema",
                                "name": "role_tags",
                                "schema": self.schema,
                                "strict": True,
                            }
                        },
                    )
    
                # Extract JSON text
                raw = getattr(response, "output_text", None)
                if not raw:
                    # fallback extractor for older objects
                    try:
                        parts = []
                        for item in getattr(response, "output", []):
                            for c in getattr(item, "content", []):
                                if getattr(c, "type", "") in ("output_text", "text"):
                                    parts.append(getattr(c, "text", ""))
                        raw = "".join(parts) if parts else None
                    except Exception:
                        raw = None
                if not raw:
                    raise RuntimeError("OpenAI response had no output_text/text content")
    
                data = json.loads(raw)
                return self._post_validate(data)
    
            except Exception as e:
                last_err = e
                if attempt == self.cfg.max_retries:
                    raise
                time.sleep(backoff)
                backoff *= 2
    
        raise RuntimeError(f"Failed after retries: {last_err}")
    
    