"""Assign personas to questions to produce the full scenario list."""

import json
import random

from loguru import logger

from prefvlm.config import cfg


def assign_scenarios() -> list[dict]:
    """Cross-product assignment: every persona × every question.

    With 5 personas, 30 questions, and assignments_per_question=5 (all personas),
    this is the full cross-product → 150 scenarios. Fixed seed for reproducibility.

    Returns list of {scenario_id, persona_id, question_id} dicts.
    """
    scenarios_path = cfg.paths.data_dir / "scenarios.json"

    if scenarios_path.exists():
        logger.info("scenarios.json already exists — loading from disk")
        with open(scenarios_path) as f:
            return json.load(f)

    questions_path = cfg.paths.data_dir / "questions.json"
    personas_path = cfg.paths.data_dir / "personas.json"

    if not questions_path.exists():
        raise FileNotFoundError("data/questions.json not found — run --stage questions first")
    if not personas_path.exists():
        raise FileNotFoundError("data/personas.json not found — run --stage personas first")

    with open(questions_path) as f:
        questions = json.load(f)
    with open(personas_path) as f:
        personas = json.load(f)

    question_ids = [q["question_id"] for q in questions]
    persona_ids = [p["persona_id"] for p in personas]

    n_per_question = cfg.scale.assignments_per_question
    if n_per_question > len(persona_ids):
        raise ValueError(
            f"assignments_per_question={n_per_question} exceeds n_personas={len(persona_ids)}"
        )

    rng = random.Random(cfg.seed)

    scenarios = []
    scenario_idx = 0

    for question_id in question_ids:
        assigned_personas = rng.sample(persona_ids, n_per_question)
        for persona_id in assigned_personas:
            scenario_idx += 1
            scenarios.append({
                "scenario_id": f"s{scenario_idx:04d}",
                "persona_id": persona_id,
                "question_id": question_id,
            })

    cfg.paths.data_dir.mkdir(parents=True, exist_ok=True)
    with open(scenarios_path, "w") as f:
        json.dump(scenarios, f, indent=2)

    logger.success(
        f"Assigned {len(scenarios)} scenarios "
        f"({len(question_ids)} questions × {n_per_question} personas each) → {scenarios_path}"
    )
    return scenarios
