#!/usr/bin/env python3

def classify_ic_manager(row):
    """
    Classifies an employee as Leadership / Manager / IC based on:
    - Mapped_L2: if "Leadership" → Leadership
    - Span: if > 0 → Manager, else → IC
    """

    l2 = str(row.get("Mapped_L2", "")).strip()
    span = int(row.get("Span", 0) or 0)

    # Rule 1: Leadership override
    if l2 == "Leadership":
        return "Leadership"

    # Rule 2: Otherwise manager vs IC based on span
    if span > 0:
        return "Manager"
    else:
        return "IC"

