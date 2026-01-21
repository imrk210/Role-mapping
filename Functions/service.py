#!/usr/bin/env python3
"""
Shared services & classes.

NOTE (current pipeline):
- batch_mapping.py uses ONLY the Rule dataclass from this module for pattern/rule hints.
- RoleTagger is currently not used by main.py/batch_mapping.py in the new batch mapping flow.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

# Optional import (so non-AI flows still work without openai installed)
try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # type: ignore


@dataclass
class TaggerConfig:
    # Kept for backward compatibility with older experiments/scripts.
    model: str = "gpt-4o-mini"
    temperature: float = 0.0
    max_retries: int = 5
    retry_backoff_seconds: float = 2.0


@dataclass
class Rule:
    """Row from the Excel `rules` sheet. Used as deterministic/heuristic hints."""
    l1: str
    l2: str
    pattern: str
    match_type: str  # equals | contains | regex
    bu: Optional[str]
    cost_center: Optional[str]
    priority: int
    notes: Optional[str]
