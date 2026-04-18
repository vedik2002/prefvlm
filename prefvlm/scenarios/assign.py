"""Assign personas to charts to produce the full scenario list."""

import json
import random

from loguru import logger

from prefvlm.config import cfg


def assign_scenarios() -> list[dict]:
    """Cross-product assignment: every persona × every chart = 100 scenarios.

    With 5 personas and 20 charts and assignments_per_chart=5 (all personas),
    this is the full cross-product. Fixed seed ensures reproducibility.

    Returns list of {scenario_id, persona_id, chart_id} dicts.
    """
    scenarios_path = cfg.paths.data_dir / "scenarios.json"

    if scenarios_path.exists():
        logger.info("scenarios.json already exists — loading from disk")
        with open(scenarios_path) as f:
            return json.load(f)

    # Load charts and personas
    charts_path = cfg.paths.data_dir / "charts.json"
    personas_path = cfg.paths.data_dir / "personas.json"

    if not charts_path.exists():
        raise FileNotFoundError("data/charts.json not found — run --stage charts first")
    if not personas_path.exists():
        raise FileNotFoundError("data/personas.json not found — run --stage personas first")

    with open(charts_path) as f:
        charts = json.load(f)
    with open(personas_path) as f:
        personas = json.load(f)

    chart_ids = [c["chart_id"] for c in charts]
    persona_ids = [p["persona_id"] for p in personas]

    n_per_chart = cfg.scale.assignments_per_chart
    if n_per_chart > len(persona_ids):
        raise ValueError(
            f"assignments_per_chart={n_per_chart} exceeds n_personas={len(persona_ids)}"
        )

    rng = random.Random(cfg.seed)

    scenarios = []
    scenario_idx = 0

    for chart_id in chart_ids:
        # Select which personas see this chart (all of them if n_per_chart == n_personas)
        assigned_personas = rng.sample(persona_ids, n_per_chart)
        for persona_id in assigned_personas:
            scenario_idx += 1
            scenarios.append({
                "scenario_id": f"s{scenario_idx:04d}",
                "persona_id": persona_id,
                "chart_id": chart_id,
            })

    cfg.paths.data_dir.mkdir(parents=True, exist_ok=True)
    with open(scenarios_path, "w") as f:
        json.dump(scenarios, f, indent=2)

    logger.success(
        f"Assigned {len(scenarios)} scenarios "
        f"({len(chart_ids)} charts × {n_per_chart} personas each) → {scenarios_path}"
    )
    return scenarios
