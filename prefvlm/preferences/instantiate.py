"""Two-step preference instantiation: dimension sampling then value assignment."""

import json
from pathlib import Path
from typing import Any, Optional

from loguru import logger
from tqdm import tqdm

from prefvlm.cache import cached_call
from prefvlm.config import cfg
from prefvlm.openai_client import chat_completion, image_message_part, load_prompt
from prefvlm.personas.schema import Persona


def _persona_dict(p: dict) -> dict:
    """Flatten persona dict for prompt formatting."""
    bf = p["big_five"]
    return {
        "name": p["name"],
        "age": p["age"],
        "occupation": p["occupation"],
        "education_level": p["education_level"],
        "data_literacy": p["data_literacy"],
        "domain_familiarity": ", ".join(p["domain_familiarity"]),
        "openness": bf["openness"],
        "conscientiousness": bf["conscientiousness"],
        "extraversion": bf["extraversion"],
        "agreeableness": bf["agreeableness"],
        "neuroticism": bf["neuroticism"],
        "hobbies": ", ".join(p["hobbies"]),
        "backstory": p["backstory"],
    }


def _sample_dimensions(
    persona: dict,
    chart: dict,
) -> list[dict]:
    """Step A: call the LLM to produce preference dimensions for this scenario."""
    prompt_template = load_prompt("preference_dimension_sampling")
    p = _persona_dict(persona)
    prompt = prompt_template.format(question=chart["question"], **p)

    image_path = Path(chart["image_path"])
    messages = [
        {
            "role": "user",
            "content": [
                image_message_part(image_path),
                {"type": "text", "text": prompt},
            ],
        }
    ]

    def _call() -> str:
        return chat_completion(
            model=cfg.models.preference_gen,
            messages=messages,
            temperature=cfg.inference.temperature_preferences,
            max_tokens=1200,
            json_mode=True,
        )

    raw = cached_call(
        "dim_sampling",
        (persona["persona_id"], chart["chart_id"]),
        _call,
    )

    try:
        parsed = json.loads(raw)
        # Model returns {"dimensions": [...]}
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
    chart: dict,
    dimensions: list[dict],
    existing_preferences: list[dict],
) -> list[dict]:
    """Step B: assign values and importances to each dimension."""
    prompt_template = load_prompt("preference_instantiation")
    p = _persona_dict(persona)

    existing_str = (
        json.dumps(existing_preferences, indent=2)
        if existing_preferences
        else "None — this is the first chart for this user."
    )
    dims_str = json.dumps(dimensions, indent=2)

    prompt = prompt_template.format(
        question=chart["question"],
        existing_preferences=existing_str,
        dimensions=dims_str,
        **p,
    )

    image_path = Path(chart["image_path"])
    messages = [
        {
            "role": "user",
            "content": [
                image_message_part(image_path),
                {"type": "text", "text": prompt},
            ],
        }
    ]

    def _call() -> str:
        return chat_completion(
            model=cfg.models.preference_gen,
            messages=messages,
            temperature=cfg.inference.temperature_preferences,
            max_tokens=1600,
            json_mode=True,
        )

    raw = cached_call(
        "dim_values",
        (persona["persona_id"], chart["chart_id"]),
        _call,
    )

    try:
        parsed = json.loads(raw)
        # Model returns {"preferences": [...]}
        values = parsed.get("preferences", parsed) if isinstance(parsed, dict) else parsed
        if not isinstance(values, list):
            raise ValueError(f"Expected list under 'preferences', got {type(values)}")
        return values
    except Exception as e:
        logger.error(f"Value instantiation parse error: {e}\nRaw: {raw[:400]}")
        raise


def _normalize_importances(instantiated: list[dict]) -> list[dict]:
    """Normalize local_importance so weights sum to 1.0."""
    total = sum(float(d.get("local_importance", 1)) for d in instantiated)
    if total == 0:
        total = len(instantiated)
    for d in instantiated:
        d["weight"] = round(float(d.get("local_importance", 1)) / total, 6)
    return instantiated


def _merge_dimensions(dimensions: list[dict], values: list[dict]) -> list[dict]:
    """Join dimension metadata with instantiated values by name."""
    dim_by_name = {d["name"]: d for d in dimensions}
    merged = []
    for v in values:
        name = v.get("name", "")
        dim = dim_by_name.get(name, {})
        merged.append({
            "name": name,
            "description": dim.get("description", ""),
            "value_range": dim.get("value_range", ""),
            "type": dim.get("type", "personal"),
            "value": v.get("value"),
            "local_importance": v.get("local_importance", 1),
            "weight": 0.0,  # filled by normalize
            "rationale": v.get("rationale", ""),
        })
    return merged


def instantiate_scenario(
    scenario: dict,
    persona: dict,
    chart: dict,
    existing_preferences: list[dict],
) -> dict:
    """Run the full two-step instantiation for one scenario.

    Returns a preference profile dict saved to data/preferences/{scenario_id}.json.
    """
    out_path = cfg.paths.preferences_dir / f"{scenario['scenario_id']}.json"
    if out_path.exists():
        logger.debug(f"Preference cache hit: {out_path.name}")
        with open(out_path) as f:
            return json.load(f)

    logger.info(
        f"Instantiating preferences for {scenario['scenario_id']} "
        f"(persona={persona['name']}, chart={chart['chart_id']})"
    )

    # Step A: dimension sampling
    dimensions = _sample_dimensions(persona, chart)
    logger.debug(f"  Sampled {len(dimensions)} dimensions")

    # Step B: value instantiation
    values = _instantiate_values(persona, chart, dimensions, existing_preferences)
    logger.debug(f"  Instantiated {len(values)} values")

    # Merge and normalize
    merged = _merge_dimensions(dimensions, values)
    merged = _normalize_importances(merged)

    profile = {
        "scenario_id": scenario["scenario_id"],
        "persona_id": persona["persona_id"],
        "chart_id": chart["chart_id"],
        "question": chart["question"],
        "preferences": merged,
    }

    cfg.paths.preferences_dir.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(profile, f, indent=2)

    logger.debug(f"  Saved to {out_path}")
    return profile


def instantiate_all(limit: int = 0) -> list[dict]:
    """Instantiate preferences for all (or first `limit`) scenarios."""
    scenarios_path = cfg.paths.data_dir / "scenarios.json"
    charts_path = cfg.paths.data_dir / "charts.json"
    personas_path = cfg.paths.data_dir / "personas.json"

    with open(scenarios_path) as f:
        scenarios = json.load(f)
    with open(charts_path) as f:
        charts_list = json.load(f)
    with open(personas_path) as f:
        personas_list = json.load(f)

    charts = {c["chart_id"]: c for c in charts_list}
    personas = {p["persona_id"]: p for p in personas_list}

    if limit:
        scenarios = scenarios[:limit]

    # Track existing preferences per persona for cross-chart consistency
    persona_existing: dict[str, list[dict]] = {pid: [] for pid in personas}

    results = []
    for scenario in tqdm(scenarios, desc="Instantiating preferences"):
        persona = personas[scenario["persona_id"]]
        chart = charts[scenario["chart_id"]]
        existing = persona_existing[scenario["persona_id"]]

        profile = instantiate_scenario(scenario, persona, chart, existing)
        results.append(profile)

        # Feed this profile's preferences as context for subsequent charts of the same persona
        if profile.get("preferences"):
            persona_existing[scenario["persona_id"]] = profile["preferences"]

    logger.success(f"Preference instantiation complete: {len(results)} scenarios")
    return results
