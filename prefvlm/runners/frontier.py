"""Frontier VLM inference (GPT-4.1-mini) for three conditions: baseline / oracle / wrong-persona."""

import json
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

from loguru import logger
from tqdm import tqdm

from prefvlm.cache import cached_call
from prefvlm.config import cfg
from prefvlm.openai_client import chat_completion, image_message_part, load_prompt


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_choices(choices: list[str]) -> str:
    labels = "ABCDEFGH"
    return "  ".join(f"{labels[i]}. {c}" for i, c in enumerate(choices))


def _format_preference_profile(prefs: list[dict], rubric: Optional[dict] = None) -> str:
    """Format a preference profile into a readable block for the oracle system prompt.

    Each line: "[name]: [value] — [plain English from rubric level 5 description]"
    Falls back to the rationale if no rubric is available.
    """
    if not prefs:
        return "(no preferences specified)"

    # Build rubric lookup: attr name → {score: description}
    rubric_levels: dict[str, dict[int, str]] = {}
    if rubric:
        for attr in rubric.get("attributes", []):
            rubric_levels[attr["name"]] = {
                lvl["score"]: lvl.get("description", "")
                for lvl in attr.get("levels", [])
            }

    lines = []
    for p in prefs:
        name = p["name"]
        value = p.get("value", "?")
        value_range = p.get("value_range", "1-5")

        # Format value display
        if isinstance(value, int) and value_range and value_range[0].isdigit():
            # Numeric: show as "2/5"
            top = value_range.split("-")[-1] if "-" in value_range else "5"
            value_display = f"{value}/{top}"
        else:
            value_display = str(value)

        # Plain English: rubric level 5 description = best match for this user's preferred value
        levels = rubric_levels.get(name, {})
        plain_english = levels.get(5, "") or p.get("rationale", "").strip()

        lines.append(f"• {name}: {value_display} — {plain_english}")

    return "\n".join(lines)


def _persona_ctx(p: dict) -> dict:
    bf = p["big_five"]
    return {
        "name": p["name"],
        "age": p["age"],
        "location": p["location"],
        "occupation": p["occupation"],
        "education_level": p["education_level"],
        "domain_familiarity": ", ".join(p.get("domain_familiarity", [])),
        "openness": bf["openness"],
        "conscientiousness": bf["conscientiousness"],
        "extraversion": bf["extraversion"],
        "agreeableness": bf["agreeableness"],
        "neuroticism": bf["neuroticism"],
        "hobbies": ", ".join(p.get("hobbies", [])),
        "backstory": p.get("backstory", ""),
    }


# ---------------------------------------------------------------------------
# Single-condition inference
# ---------------------------------------------------------------------------

def _run_one(
    scenario: dict,
    question: dict,
    persona: dict,
    preference_profile: list[dict],
    condition: str,
    model: str,
    rubric: Optional[dict] = None,
) -> dict:
    """Run one (scenario, condition) inference. Returns response dict."""
    out_path = (
        cfg.paths.responses_dir / "frontier" / f"{scenario['scenario_id']}_{condition}.json"
    )
    if out_path.exists():
        logger.debug(f"Response cache hit: {out_path.name}")
        with open(out_path) as f:
            return json.load(f)

    image_path = Path(question["image_path"])
    img_part = image_message_part(image_path)
    choices_str = _format_choices(question["choices"])

    # ---- Build messages ----
    if condition == "baseline":
        user_tmpl = load_prompt("baseline_user")
        user_text = user_tmpl.format(
            question=question["question"],
            choices=choices_str,
        )
        messages = [
            {"role": "user", "content": [img_part, {"type": "text", "text": user_text}]},
        ]

    elif condition in ("oracle", "wrong_persona"):
        system_tmpl = load_prompt("oracle_system")
        user_tmpl = load_prompt("oracle_user")

        ctx = _persona_ctx(persona)
        pref_str = _format_preference_profile(preference_profile, rubric=rubric)
        system_text = system_tmpl.format(
            preference_profile=pref_str,
            **ctx,
        )
        user_text = user_tmpl.format(
            question=question["question"],
            choices=choices_str,
        )
        messages = [
            {"role": "system", "content": system_text},
            {"role": "user", "content": [img_part, {"type": "text", "text": user_text}]},
        ]
    else:
        raise ValueError(f"Unknown condition: {condition!r}")

    # ---- Call API (cached) ----
    response_text = cached_call(
        "frontier_response",
        (scenario["scenario_id"], condition, model),
        lambda: chat_completion(
            model=model,
            messages=messages,
            temperature=cfg.inference.temperature_responses,
            max_tokens=cfg.inference.max_tokens,
        ),
    )

    result = {
        "scenario_id": scenario["scenario_id"],
        "question_id": scenario["question_id"],
        "persona_id": scenario["persona_id"],
        "condition": condition,
        "model": model,
        "persona_name": persona["name"],
        "response": response_text,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    logger.debug(f"Saved response → {out_path.name}")
    return result


# ---------------------------------------------------------------------------
# Wrong-persona selector
# ---------------------------------------------------------------------------

def _pick_wrong_persona_prefs(
    scenario: dict,
    all_scenarios: list[dict],
    preferences_dir: Path,
    personas: dict[str, dict],
    seed: int,
) -> tuple[dict, list[dict]]:
    """Return (wrong_persona, wrong_prefs) for the wrong-persona condition.

    Selects the maximally distant persona on the same question: picks the
    candidate whose top-3 preference values differ most from the own persona's.
    Tie-break: largest difference on the single top-weight attribute.
    Falls back to random if no preference files are loaded.
    """
    same_question = [
        s for s in all_scenarios
        if s["question_id"] == scenario["question_id"]
        and s["persona_id"] != scenario["persona_id"]
    ]
    if not same_question:
        same_question = [s for s in all_scenarios if s["scenario_id"] != scenario["scenario_id"]]

    # Load own preferences; sort by local_importance descending
    own_pref_path = preferences_dir / f"{scenario['scenario_id']}.json"
    own_prefs: list[dict] = []
    if own_pref_path.exists():
        with open(own_pref_path) as f:
            own_prefs = json.load(f).get("preferences", [])
    def _to_float(v, default=3.0):
        try:
            return float(v)
        except (TypeError, ValueError):
            return default

    own_sorted = sorted(own_prefs, key=lambda p: _to_float(p.get("local_importance", 0)), reverse=True)
    # Only include attributes with numeric values in the distance computation
    own_numeric = [p for p in own_sorted if isinstance(p.get("value"), (int, float))]
    top3_own = own_numeric[:3] if own_numeric else own_sorted[:3]
    top3_names = [p["name"] for p in top3_own]
    top3_values = {p["name"]: _to_float(p.get("value", 3)) for p in top3_own}

    best_scenario = None
    best_avg_diff = -1.0
    best_top1_diff = -1.0

    for cand in same_question:
        cand_path = preferences_dir / f"{cand['scenario_id']}.json"
        if not cand_path.exists():
            continue
        with open(cand_path) as f:
            cand_prefs = json.load(f).get("preferences", [])
        cand_by_name = {p["name"]: _to_float(p.get("value", 3)) for p in cand_prefs}

        diffs = []
        for name in top3_names:
            if name in cand_by_name:
                diffs.append(abs(top3_values[name] - cand_by_name[name]))

        if not diffs:
            continue

        avg_diff = sum(diffs) / len(diffs)
        top1_diff = diffs[0] if diffs else 0.0

        if avg_diff > best_avg_diff or (avg_diff == best_avg_diff and top1_diff > best_top1_diff):
            best_avg_diff = avg_diff
            best_top1_diff = top1_diff
            best_scenario = cand
            best_prefs = cand_prefs

    if best_scenario is None:
        # Fallback to deterministic random if no pref files found
        rng = random.Random(seed + hash(scenario["scenario_id"]))
        best_scenario = rng.choice(same_question)
        pref_path = preferences_dir / f"{best_scenario['scenario_id']}.json"
        best_prefs = []
        if pref_path.exists():
            with open(pref_path) as f:
                best_prefs = json.load(f).get("preferences", [])

    wrong_persona = personas[best_scenario["persona_id"]]
    logger.debug(
        f"  wrong-persona: {wrong_persona['name']} (scenario {best_scenario['scenario_id']}, "
        f"avg_diff={best_avg_diff:.2f})"
    )
    return wrong_persona, best_prefs


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------

def _frontier_scenario_worker(
    scenario: dict,
    all_scenarios: list[dict],
    questions: dict,
    personas: dict,
    model: str,
    seed: int,
) -> list[dict]:
    """Run all 3 conditions for one scenario. Returns list of response dicts."""
    conditions = ["baseline", "oracle", "wrong_persona"]
    question = questions[scenario["question_id"]]
    persona = personas[scenario["persona_id"]]

    pref_path = cfg.paths.preferences_dir / f"{scenario['scenario_id']}.json"
    if pref_path.exists():
        with open(pref_path) as f:
            own_prefs = json.load(f).get("preferences", [])
    else:
        logger.warning(f"No preference file for {scenario['scenario_id']}, using empty prefs")
        own_prefs = []

    rubric_path = cfg.paths.rubrics_dir / f"{scenario['scenario_id']}.json"
    own_rubric = None
    if rubric_path.exists():
        with open(rubric_path) as f:
            own_rubric = json.load(f)

    results = []
    for condition in conditions:
        if condition == "wrong_persona":
            wp, wp_prefs = _pick_wrong_persona_prefs(
                scenario, all_scenarios, cfg.paths.preferences_dir, personas, seed
            )
            inf_persona = wp
            inf_prefs = wp_prefs
            inf_rubric = own_rubric
        else:
            inf_persona = persona
            inf_prefs = own_prefs
            inf_rubric = own_rubric

        try:
            resp = _run_one(scenario, question, inf_persona, inf_prefs, condition, model,
                            rubric=inf_rubric)
            results.append(resp)
            logger.info(
                f"[{scenario['scenario_id']}] {condition:12s} → {len(resp['response'])} chars"
            )
        except Exception as e:
            logger.error(f"[{scenario['scenario_id']}] {condition} FAILED: {e}")

    return results


def run_frontier_inference(limit: int = 0, workers: int = 20) -> list[dict]:
    """Run GPT-4.1-mini inference for baseline / oracle / wrong-persona conditions.

    Parallelises across scenarios — all 3 conditions per scenario run sequentially
    within each worker thread.

    Args:
        limit: Process only first N scenarios (0 = all).
        workers: Number of parallel threads.

    Returns:
        List of response dicts (one per scenario × condition).
    """
    with open(cfg.paths.data_dir / "scenarios.json") as f:
        all_scenarios = json.load(f)
    with open(cfg.paths.data_dir / "questions.json") as f:
        questions_list = json.load(f)
    with open(cfg.paths.data_dir / "personas.json") as f:
        personas_list = json.load(f)

    questions = {q["question_id"]: q for q in questions_list}
    personas = {p["persona_id"]: p for p in personas_list}

    scenarios = all_scenarios[:limit] if limit else all_scenarios
    model = cfg.models.tested_frontier
    seed = cfg.seed

    results = []
    with tqdm(total=len(scenarios), desc="Frontier inference") as pbar:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(
                    _frontier_scenario_worker,
                    s, all_scenarios, questions, personas, model, seed
                ): s["scenario_id"]
                for s in scenarios
            }
            for future in as_completed(futures):
                sid = futures[future]
                try:
                    resps = future.result()
                    results.extend(resps)
                except Exception as e:
                    logger.error(f"[{sid}] scenario worker FAILED: {e}")
                pbar.update(1)

    logger.success(f"Frontier inference complete: {len(results)} responses")
    return results
