"""Two-step preference instantiation: dimension sampling then value assignment."""

import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from loguru import logger
from tqdm import tqdm

from prefvlm.cache import cached_call
from prefvlm.config import cfg
from prefvlm.openai_client import chat_completion, image_message_part, load_prompt


def _persona_ctx(p: dict) -> dict:
    """Flatten persona fields for prompt formatting."""
    bf = p["big_five"]
    big_five_str = (
        f"openness={bf['openness']}, conscientiousness={bf['conscientiousness']}, "
        f"extraversion={bf['extraversion']}, agreeableness={bf['agreeableness']}, "
        f"neuroticism={bf['neuroticism']}"
    )
    return {
        "name": p["name"],
        "age": p["age"],
        "occupation": p["occupation"],
        "education_level": p["education_level"],
        "domain_familiarity": ", ".join(p["domain_familiarity"]),
        "openness": bf["openness"],
        "conscientiousness": bf["conscientiousness"],
        "extraversion": bf["extraversion"],
        "agreeableness": bf["agreeableness"],
        "neuroticism": bf["neuroticism"],
        "big_five": big_five_str,
        "hobbies": ", ".join(p["hobbies"]),
        "backstory": p["backstory"],
    }


def _format_choices(choices: list[str]) -> str:
    labels = "ABCDEFGH"
    return "  ".join(f"{labels[i]}. {c}" for i, c in enumerate(choices))


def _sample_dimensions(persona: dict, question: dict) -> list[dict]:
    """Step A: produce preference dimensions for this persona × question."""
    prompt_template = load_prompt("preference_dimension_sampling")
    p = _persona_ctx(persona)
    choices_str = _format_choices(question.get("choices", []))
    prompt = prompt_template.format(
        question=question["question"],
        choices=choices_str,
        topic=question.get("topic", ""),
        grade=question.get("grade", ""),
        **p,
    )

    image_path = Path(question["image_path"])
    messages = [
        {
            "role": "user",
            "content": [
                image_message_part(image_path),
                {"type": "text", "text": prompt},
            ],
        }
    ]

    raw = cached_call(
        "dim_sampling",
        (persona["persona_id"], question["question_id"]),
        lambda: chat_completion(
            model=cfg.models.preference_gen,
            messages=messages,
            temperature=cfg.inference.temperature_preferences,
            max_tokens=1400,
            json_mode=True,
        ),
    )

    try:
        parsed = json.loads(raw)
        dims = parsed.get("dimensions", parsed) if isinstance(parsed, dict) else parsed
        if not isinstance(dims, list):
            raise ValueError(f"Expected list under 'dimensions', got {type(dims)}")
        for d in dims:
            for k in ("name", "description", "value_range", "type"):
                if k not in d:
                    raise ValueError(f"Dimension missing key '{k}': {d}")
        return dims
    except Exception as e:
        logger.error(f"Dimension sampling parse error: {e}\nRaw: {raw[:400]}")
        raise


def _instantiate_values(
    persona: dict,
    question: dict,
    dimensions: list[dict],
    existing_preferences: list[dict],
) -> list[dict]:
    """Step B: assign values and importances to each dimension."""
    prompt_template = load_prompt("preference_instantiation")
    p = _persona_ctx(persona)
    choices_str = _format_choices(question.get("choices", []))

    existing_str = (
        json.dumps(existing_preferences, indent=2)
        if existing_preferences
        else "None — this is the first question for this user."
    )
    dims_str = json.dumps(dimensions, indent=2)

    prompt = prompt_template.format(
        question=question["question"],
        choices=choices_str,
        topic=question.get("topic", ""),
        grade=question.get("grade", ""),
        existing_preferences=existing_str,
        dimensions=dims_str,
        **p,
    )

    image_path = Path(question["image_path"])
    messages = [
        {
            "role": "user",
            "content": [
                image_message_part(image_path),
                {"type": "text", "text": prompt},
            ],
        }
    ]

    raw = cached_call(
        "dim_values",
        (persona["persona_id"], question["question_id"]),
        lambda: chat_completion(
            model=cfg.models.preference_gen,
            messages=messages,
            temperature=cfg.inference.temperature_preferences,
            max_tokens=1800,
            json_mode=True,
        ),
    )

    try:
        parsed = json.loads(raw)
        values = parsed.get("preferences", parsed) if isinstance(parsed, dict) else parsed
        if not isinstance(values, list):
            raise ValueError(f"Expected list under 'preferences', got {type(values)}")
        return values
    except Exception as e:
        logger.error(f"Value instantiation parse error: {e}\nRaw: {raw[:400]}")
        raise


def _normalize_importances(instantiated: list[dict]) -> list[dict]:
    total = sum(float(d.get("local_importance", 1)) for d in instantiated)
    if total == 0:
        total = len(instantiated)
    for d in instantiated:
        d["weight"] = round(float(d.get("local_importance", 1)) / total, 6)
    return instantiated


# Value ranges for Big Five–derived attributes (matches the prompt spec)
_BIG_FIVE_ATTR_RANGES: dict[str, str] = {
    "Reassuring Framing": "none|some|strong",
    "Explanation Structure": "implicit|partial|explicit",
    "Conversational Warmth": "formal|neutral|warm",
    "Encouragement Tone": "direct|balanced|elaborate",
    "Creative Framing": "conventional|some|frequent",
}

_MAX_ATTRIBUTES = 15
_MIN_ATTRIBUTES = 12

def _merge(dimensions: list[dict], values: list[dict]) -> list[dict]:
    dim_by_name = {d["name"]: d for d in dimensions}
    merged = []
    for v in values:
        name = v.get("name", "")
        dim = dim_by_name.get(name, {})
        # Fall back to Big Five spec ranges for attributes not in step-A dimensions
        value_range = dim.get("value_range", "") or _BIG_FIVE_ATTR_RANGES.get(name, "")
        merged.append({
            "name": name,
            "description": dim.get("description", ""),
            "value_range": value_range,
            "type": dim.get("type", "personal"),
            "value": v.get("value"),
            "local_importance": v.get("local_importance", 1),
            "weight": 0.0,
            "rationale": v.get("rationale", ""),
        })

    # Hard-enforce 15-attribute ceiling: drop lowest local_importance first
    if len(merged) > _MAX_ATTRIBUTES:
        merged.sort(key=lambda x: x["local_importance"], reverse=True)
        merged = merged[:_MAX_ATTRIBUTES]
        logger.debug(f"Trimmed attributes to {_MAX_ATTRIBUTES} (was {len(values)})")

    return merged


def instantiate_scenario(
    scenario: dict,
    persona: dict,
    question: dict,
    existing_preferences: list[dict],
) -> dict:
    """Full two-step instantiation for one scenario → data/preferences/{scenario_id}.json."""
    out_path = cfg.paths.preferences_dir / f"{scenario['scenario_id']}.json"
    if out_path.exists():
        logger.debug(f"Preference cache hit: {out_path.name}")
        with open(out_path) as f:
            return json.load(f)

    logger.info(
        f"Instantiating {scenario['scenario_id']} "
        f"(persona={persona['name']}, question={question['question_id']})"
    )

    dimensions = _sample_dimensions(persona, question)
    logger.debug(f"  {len(dimensions)} dimensions sampled")

    values = _instantiate_values(persona, question, dimensions, existing_preferences)
    logger.debug(f"  {len(values)} values instantiated")

    merged = _normalize_importances(_merge(dimensions, values))

    profile = {
        "scenario_id": scenario["scenario_id"],
        "persona_id": persona["persona_id"],
        "question_id": question["question_id"],
        "question": question["question"],
        "choices": question.get("choices", []),
        "answer_index": question.get("answer_index"),
        "topic": question.get("topic", ""),
        "grade": question.get("grade"),
        "preferences": merged,
    }

    cfg.paths.preferences_dir.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(profile, f, indent=2)
    return profile


def _instantiate_persona_group(
    persona: dict,
    persona_scenarios: list[dict],
    questions: dict,
) -> list[dict]:
    """Process all scenarios for one persona sequentially (preserving existing context)."""
    existing: list[dict] = []
    results = []
    for scenario in persona_scenarios:
        question = questions[scenario["question_id"]]
        profile = instantiate_scenario(scenario, persona, question, existing)
        results.append(profile)
        if profile.get("preferences"):
            existing = profile["preferences"]
    return results


def instantiate_all(limit: int = 0, workers: int = 20) -> list[dict]:
    """Instantiate preferences for all (or first `limit`) scenarios.

    Parallelises across personas — scenarios within each persona remain
    sequential so the existing-preference context chain is preserved.
    """
    with open(cfg.paths.data_dir / "scenarios.json") as f:
        scenarios = json.load(f)
    with open(cfg.paths.data_dir / "questions.json") as f:
        questions_list = json.load(f)
    with open(cfg.paths.data_dir / "personas.json") as f:
        personas_list = json.load(f)

    questions = {q["question_id"]: q for q in questions_list}
    personas = {p["persona_id"]: p for p in personas_list}

    if limit:
        scenarios = scenarios[:limit]

    # Group scenarios by persona, preserving assignment order
    by_persona: dict[str, list[dict]] = defaultdict(list)
    for s in scenarios:
        by_persona[s["persona_id"]].append(s)

    total = len(scenarios)
    results_map: dict[str, dict] = {}

    with tqdm(total=total, desc="Instantiating preferences") as pbar:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(
                    _instantiate_persona_group,
                    personas[pid],
                    pscenarios,
                    questions,
                ): pid
                for pid, pscenarios in by_persona.items()
            }
            for future in as_completed(futures):
                pid = futures[future]
                try:
                    group_results = future.result()
                    for r in group_results:
                        results_map[r["scenario_id"]] = r
                    pbar.update(len(group_results))
                except Exception as e:
                    logger.error(f"Persona {pid} group failed: {e}")

    # Return in original scenario order
    results = [results_map[s["scenario_id"]] for s in scenarios if s["scenario_id"] in results_map]
    logger.success(f"Preference instantiation complete: {len(results)} scenarios")
    return results
