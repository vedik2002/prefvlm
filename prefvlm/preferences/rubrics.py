"""Generate per-attribute scoring rubrics calibrated to each persona's preference profile."""

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from loguru import logger
from tqdm import tqdm

from prefvlm.cache import cached_call
from prefvlm.config import cfg
from prefvlm.openai_client import chat_completion, load_prompt


def _persona_ctx(p: dict) -> dict:
    bf = p["big_five"]
    return {
        "persona_name": p["name"],
        "occupation": p["occupation"],
        "education_level": p["education_level"],
        "openness": bf["openness"],
        "conscientiousness": bf["conscientiousness"],
        "extraversion": bf["extraversion"],
        "agreeableness": bf["agreeableness"],
        "neuroticism": bf["neuroticism"],
        "hobbies": ", ".join(p.get("hobbies", [])),
    }


def _generate_attribute_rubric(
    dim: dict,
    persona: dict,
) -> list[dict]:
    """Generate a 5-level rubric for one preference dimension.

    Uses rubric_expertise.txt for type='expertise', rubric_personal.txt for type='personal'.
    Returns list of 5 level dicts: [{score, label, description}, ...].
    """
    dim_type = dim.get("type", "personal")
    prompt_name = "rubric_expertise" if dim_type == "expertise" else "rubric_personal"
    prompt_template = load_prompt(prompt_name)

    ctx = _persona_ctx(persona)
    prompt = prompt_template.format(
        name=dim["name"],
        description=dim.get("description", ""),
        value_range=dim.get("value_range", "1-5"),
        value=dim.get("value", 3),
        rationale=dim.get("rationale", ""),
        **ctx,
    )

    raw = cached_call(
        "rubric",
        (persona["persona_id"], dim["name"], str(dim.get("value"))),
        lambda: chat_completion(
            model=cfg.models.rubric_gen,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=800,
            json_mode=True,
        ),
    )

    try:
        parsed = json.loads(raw)
        levels = parsed.get("rubric", parsed) if isinstance(parsed, dict) else parsed
        if not isinstance(levels, list):
            raise ValueError(f"Expected list under 'rubric', got {type(levels)}")
        # Validate: must have exactly 5 levels with score 1-5
        if len(levels) != 5:
            raise ValueError(f"Expected 5 rubric levels, got {len(levels)}")
        return levels
    except Exception as e:
        logger.error(f"Rubric parse error for '{dim['name']}': {e}\nRaw: {raw[:300]}")
        raise


def generate_scenario_rubric(
    scenario: dict,
    persona: dict,
    preference_profile: list[dict],
) -> dict:
    """Generate rubrics for all dimensions in a scenario's preference profile.

    Returns a rubric dict saved to data/rubrics/{scenario_id}.json.
    """
    out_path = cfg.paths.rubrics_dir / f"{scenario['scenario_id']}.json"
    if out_path.exists():
        logger.debug(f"Rubric cache hit: {out_path.name}")
        with open(out_path) as f:
            return json.load(f)

    logger.info(
        f"Generating rubric for {scenario['scenario_id']} "
        f"(persona={persona['name']}, {len(preference_profile)} dimensions)"
    )

    rubric_attrs = []
    for dim in preference_profile:
        try:
            levels = _generate_attribute_rubric(dim, persona)
            rubric_attrs.append({
                "name": dim["name"],
                "type": dim.get("type", "personal"),
                "weight": dim.get("weight", 0.0),
                "preferred_value": dim.get("value"),
                "levels": levels,
            })
        except Exception as e:
            logger.warning(f"  Skipping dim '{dim['name']}' due to error: {e}")
            continue

    rubric = {
        "scenario_id": scenario["scenario_id"],
        "persona_id": persona["persona_id"],
        "question_id": scenario.get("question_id"),
        "n_attributes": len(rubric_attrs),
        "attributes": rubric_attrs,
    }

    cfg.paths.rubrics_dir.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(rubric, f, indent=2)
    logger.debug(f"  Saved rubric with {len(rubric_attrs)} attributes → {out_path}")
    return rubric


def _rubric_worker(scenario: dict, persona: dict) -> dict | None:
    """Worker: load preference file and generate rubric for one scenario."""
    pref_path = cfg.paths.preferences_dir / f"{scenario['scenario_id']}.json"
    if not pref_path.exists():
        logger.warning(f"No preference file for {scenario['scenario_id']}, skipping")
        return None
    with open(pref_path) as f:
        pref_profile = json.load(f)
    return generate_scenario_rubric(
        scenario, persona, pref_profile.get("preferences", [])
    )


def generate_all_rubrics(limit: int = 0, workers: int = 20) -> list[dict]:
    """Generate rubrics for all (or first `limit`) scenarios in parallel."""
    with open(cfg.paths.data_dir / "scenarios.json") as f:
        scenarios = json.load(f)
    with open(cfg.paths.data_dir / "personas.json") as f:
        personas_list = json.load(f)

    personas = {p["persona_id"]: p for p in personas_list}

    if limit:
        scenarios = scenarios[:limit]

    results = []
    with tqdm(total=len(scenarios), desc="Generating rubrics") as pbar:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(_rubric_worker, s, personas[s["persona_id"]]): s["scenario_id"]
                for s in scenarios
            }
            for future in as_completed(futures):
                sid = futures[future]
                try:
                    rubric = future.result()
                    if rubric:
                        results.append(rubric)
                except Exception as e:
                    logger.error(f"Rubric failed for {sid}: {e}")
                pbar.update(1)

    logger.success(f"Rubric generation complete: {len(results)} scenarios")
    return results
