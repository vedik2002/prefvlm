"""
PrefVLM-MVP Offline Diagnostic Checks
Checks A, B, C, D — read-only, no API calls.
"""

import json
import os
import random
import re
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE = Path("/Users/vedik/prefvlm-mvp/data")
SCENARIOS_FILE  = BASE / "scenarios.json"
PERSONAS_FILE   = BASE / "personas.json"
PREFERENCES_DIR = BASE / "preferences"
RUBRICS_DIR     = BASE / "rubrics"
SEED = 42

# ── Load master data ──────────────────────────────────────────────────────────
scenarios = json.loads(SCENARIOS_FILE.read_text())
personas_list = json.loads(PERSONAS_FILE.read_text())
personas = {p["persona_id"]: p for p in personas_list}

def load_prefs(scenario_id: str) -> dict:
    path = PREFERENCES_DIR / f"{scenario_id}.json"
    return json.loads(path.read_text()) if path.exists() else {}

def load_rubric(scenario_id: str) -> dict:
    path = RUBRICS_DIR / f"{scenario_id}.json"
    return json.loads(path.read_text()) if path.exists() else {}

# ── Keyword helpers ───────────────────────────────────────────────────────────
EXPERTISE_KEYWORDS = re.compile(r"terminology|technical|expertise|jargon|complexity", re.I)
TONE_KEYWORDS      = re.compile(r"tone|reassur|warm|encouragement", re.I)
POLARITY_KEYWORDS  = re.compile(r"terminology|technical|expertise", re.I)

def find_expertise_attr(preferences: list) -> dict | None:
    """Return best-matching expertise/terminology preference (numeric value only)."""
    for pref in preferences:
        if EXPERTISE_KEYWORDS.search(pref.get("name", "")) and isinstance(pref.get("value"), (int, float)):
            return pref
    return None

def find_tone_attrs(preferences: list) -> list:
    """Return all tone/reassurance preferences (numeric value only)."""
    return [
        p for p in preferences
        if TONE_KEYWORDS.search(p.get("name", "")) and isinstance(p.get("value"), (int, float))
    ]

# ── Pretty table helpers ──────────────────────────────────────────────────────
def col_widths(headers: list[str], rows: list[list]) -> list[int]:
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))
    return widths

def print_table(headers: list[str], rows: list[list]) -> None:
    widths = col_widths(headers, rows)
    sep = "+-" + "-+-".join("-" * w for w in widths) + "-+"
    hdr = "| " + " | ".join(str(h).ljust(widths[i]) for i, h in enumerate(headers)) + " |"
    print(sep)
    print(hdr)
    print(sep)
    for row in rows:
        line = "| " + " | ".join(str(c).ljust(widths[i]) for i, c in enumerate(row)) + " |"
        print(line)
    print(sep)

# ══════════════════════════════════════════════════════════════════════════════
# CHECK A — Education vs expertise alignment
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("CHECK A — Education vs Expertise Alignment")
print("=" * 80)

HIGH_EDU = {"master's degree", "doctoral degree", "professional degree"}
LOW_EDU  = {"high school diploma", "vocational or trade school certificate",
            "some college, no degree", "vocational training", "some college"}

check_a_flags = []   # (scenario_id, reason)
rows_a = []

for sc in scenarios:
    sid = sc["scenario_id"]
    pid = sc["persona_id"]
    persona = personas.get(pid, {})
    edu = persona.get("education_level", "unknown")

    pref_data = load_prefs(sid)
    prefs = pref_data.get("preferences", [])
    attr = find_expertise_attr(prefs)

    if attr is None:
        attr_name  = "(none found)"
        attr_value = "N/A"
        flag = ""
    else:
        attr_name  = attr["name"]
        attr_value = attr["value"]

        flag = ""
        if edu in HIGH_EDU and isinstance(attr_value, (int, float)) and attr_value < 3:
            flag = "★ high-edu low-expertise"
            check_a_flags.append((sid, f"edu={edu!r} attr={attr_name!r} val={attr_value}"))
        elif edu in LOW_EDU and isinstance(attr_value, (int, float)) and attr_value > 2:
            flag = "★ low-edu high-expertise"
            check_a_flags.append((sid, f"edu={edu!r} attr={attr_name!r} val={attr_value}"))

    rows_a.append([sid, pid, edu, attr_name, attr_value, flag])

# Sort by education_level
rows_a.sort(key=lambda r: r[2])

headers_a = ["scenario_id", "persona_id", "education_level", "attr_name", "value", "flag"]
print_table(headers_a, rows_a)

print(f"\nCheck A: {len(check_a_flags)} flag(s)")
for sid, reason in check_a_flags:
    print(f"  {sid}: {reason}")

# ══════════════════════════════════════════════════════════════════════════════
# CHECK B — Neuroticism vs tone/reassurance
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("CHECK B — Neuroticism vs Tone/Reassurance")
print("=" * 80)

check_b_flags = []
rows_b = []

for sc in scenarios:
    sid = sc["scenario_id"]
    pid = sc["persona_id"]
    persona = personas.get(pid, {})
    neuroticism = persona.get("big_five", {}).get("neuroticism", "unknown")

    pref_data = load_prefs(sid)
    prefs = pref_data.get("preferences", [])
    tone_attrs = find_tone_attrs(prefs)

    if not tone_attrs:
        attr_summary = "(none found)"
        flag = ""
    else:
        attr_summary = "; ".join(f"{a['name']}={a['value']}" for a in tone_attrs)
        values = [a["value"] for a in tone_attrs if isinstance(a["value"], (int, float))]
        flag = ""
        if values:
            if neuroticism == "high" and all(v <= 2 for v in values):
                flag = "★ high-neuro all-low-tone"
                check_b_flags.append((sid, f"neuroticism=high, tone attrs all <=2: {attr_summary}"))
            elif neuroticism == "low" and all(v >= 4 for v in values):
                flag = "★ low-neuro all-high-tone"
                check_b_flags.append((sid, f"neuroticism=low, tone attrs all >=4: {attr_summary}"))

    rows_b.append([sid, pid, neuroticism, attr_summary, flag])

rows_b.sort(key=lambda r: {"low": 0, "moderate": 1, "high": 2, "unknown": 3}.get(r[2], 3))

headers_b = ["scenario_id", "persona_id", "neuroticism", "tone_attrs (name=value)", "flag"]
print_table(headers_b, rows_b)

print(f"\nCheck B: {len(check_b_flags)} flag(s)")
for sid, reason in check_b_flags:
    print(f"  {sid}: {reason}")

# ══════════════════════════════════════════════════════════════════════════════
# CHECK C — Rubric polarity (preferred_value vs score-5 description)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("CHECK C — Rubric Polarity (preferred_value vs score-5 description)")
print("=" * 80)

PLAIN_WORDS    = re.compile(r"\bplain\b|\bsimple\b|\bminimal\b|\beveryday\b|\bbasic\b", re.I)
TECHNICAL_WORDS = re.compile(r"\btechnical\b|\bexpert\b|\badvanced\b|\bspecialist\b", re.I)

check_c_flags = []
rows_c = []

for sc in scenarios:
    sid = sc["scenario_id"]
    rubric = load_rubric(sid)
    if not rubric:
        continue
    for attr in rubric.get("attributes", []):
        if not POLARITY_KEYWORDS.search(attr.get("name", "")):
            continue
        preferred_value = attr.get("preferred_value")
        if not isinstance(preferred_value, (int, float)):
            continue
        levels = attr.get("levels", [])
        score5 = next((lv for lv in levels if lv.get("score") == 5), None)
        if score5 is None:
            continue
        desc5 = score5.get("description", "")
        desc5_short = desc5[:120].replace("\n", " ")

        flag = ""
        if preferred_value >= 4 and PLAIN_WORDS.search(desc5):
            flag = "★ high-pref but plain score-5"
            check_c_flags.append((sid, f"attr={attr['name']!r} pref={preferred_value}, score-5 mentions 'plain/simple/...'"))
        elif preferred_value <= 2 and TECHNICAL_WORDS.search(desc5):
            flag = "★ low-pref but technical score-5"
            check_c_flags.append((sid, f"attr={attr['name']!r} pref={preferred_value}, score-5 mentions 'technical/expert/...'"))

        rows_c.append([sid, attr["name"], preferred_value, desc5_short, flag])

headers_c = ["scenario_id", "attr_name", "preferred_value", "score_5_description (truncated)", "flag"]
print_table(headers_c, rows_c)

print(f"\nCheck C: {len(check_c_flags)} flag(s)")
for sid, reason in check_c_flags:
    print(f"  {sid}: {reason}")

# ══════════════════════════════════════════════════════════════════════════════
# CHECK D — Wrong-persona distance
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("CHECK D — Wrong-Persona Distance")
print("=" * 80)

# Build lookup: question_id -> list of scenarios
from collections import defaultdict
by_question: dict[str, list[dict]] = defaultdict(list)
for sc in scenarios:
    by_question[sc["question_id"]].append(sc)

check_d_flags = []
rows_d = []

for sc in scenarios:
    sid      = sc["scenario_id"]
    pid      = sc["persona_id"]
    qid      = sc["question_id"]

    # Determine wrong persona
    candidates = [s for s in by_question[qid] if s["persona_id"] != pid]
    if not candidates:
        rows_d.append([sid, pid, "(no candidates)", "N/A", "N/A", ""])
        continue

    rng = random.Random(SEED + hash(sid))
    wrong_sc   = rng.choice(candidates)
    wrong_pid  = wrong_sc["persona_id"]
    wrong_sid  = wrong_sc["scenario_id"]

    # Load own preferences sorted by local_importance desc
    own_prefs = load_prefs(sid).get("preferences", [])
    own_numeric = [p for p in own_prefs if isinstance(p.get("value"), (int, float))]
    own_numeric.sort(key=lambda p: p.get("local_importance", 0), reverse=True)
    top3 = own_numeric[:3]

    # Load wrong persona's preferences for the SAME question (their scenario)
    wrong_prefs = load_prefs(wrong_sid).get("preferences", [])
    wrong_map = {p["name"]: p["value"] for p in wrong_prefs if isinstance(p.get("value"), (int, float))}

    diffs = []
    for attr in top3:
        name = attr["name"]
        own_val = attr["value"]
        if name in wrong_map:
            diffs.append(abs(own_val - wrong_map[name]))

    if diffs:
        avg_diff = sum(diffs) / len(diffs)
    else:
        avg_diff = None

    top3_names = "; ".join(a["name"] for a in top3)
    flag = ""
    if avg_diff is not None and avg_diff < 1.0:
        flag = "★ avg_diff < 1.0"
        check_d_flags.append((sid, f"own={pid} wrong={wrong_pid} avg_diff={avg_diff:.3f} top3=[{top3_names}]"))

    avg_str = f"{avg_diff:.3f}" if avg_diff is not None else "N/A"
    rows_d.append([sid, pid, wrong_pid, avg_str, top3_names, flag])

headers_d = ["scenario_id", "own_persona", "wrong_persona", "avg_diff", "top3_attr_names", "flag"]
print_table(headers_d, rows_d)

print(f"\nCheck D: {len(check_d_flags)} flag(s)")
for sid, reason in check_d_flags:
    print(f"  {sid}: {reason}")

# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

total_scenarios = len(scenarios)

a_ids = set(x[0] for x in check_a_flags)
b_ids = set(x[0] for x in check_b_flags)
c_ids = set(x[0] for x in check_c_flags)
d_ids = set(x[0] for x in check_d_flags)

print(f"\nTotal scenarios checked : {total_scenarios}")

print(f"\nCheck A failures ({len(a_ids)} scenarios, {len(check_a_flags)} flags):")
if a_ids:
    for sid in sorted(a_ids):
        print(f"  {sid}")
else:
    print("  (none)")

print(f"\nCheck B failures ({len(b_ids)} scenarios, {len(check_b_flags)} flags):")
if b_ids:
    for sid in sorted(b_ids):
        print(f"  {sid}")
else:
    print("  (none)")

print(f"\nCheck C failures ({len(c_ids)} scenarios, {len(check_c_flags)} flags):")
if c_ids:
    for sid in sorted(c_ids):
        print(f"  {sid}")
else:
    print("  (none)")

print(f"\nCheck D failures ({len(d_ids)} scenarios, {len(check_d_flags)} flags):")
if d_ids:
    for sid in sorted(d_ids):
        print(f"  {sid}")
else:
    print("  (none)")

# Scenarios failing 2+ checks
all_failed: dict[str, int] = defaultdict(int)
for sid in a_ids: all_failed[sid] += 1
for sid in b_ids: all_failed[sid] += 1
for sid in c_ids: all_failed[sid] += 1
for sid in d_ids: all_failed[sid] += 1
multi_fail = sorted(sid for sid, cnt in all_failed.items() if cnt >= 2)

print(f"\nScenarios failing 2+ checks ({len(multi_fail)}):")
if multi_fail:
    for sid in multi_fail:
        checks = []
        if sid in a_ids: checks.append("A")
        if sid in b_ids: checks.append("B")
        if sid in c_ids: checks.append("C")
        if sid in d_ids: checks.append("D")
        print(f"  {sid}: checks {', '.join(checks)}")
else:
    print("  (none)")

# Which check has most failures
counts = {"A": len(a_ids), "B": len(b_ids), "C": len(c_ids), "D": len(d_ids)}
worst = max(counts, key=lambda k: counts[k])
print(f"\nCheck with most failures: Check {worst} ({counts[worst]} scenarios)")

print("\n" + "=" * 80)
print("Done.")
print("=" * 80)
