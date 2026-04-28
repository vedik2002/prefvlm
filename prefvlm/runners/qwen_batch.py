"""Qwen batch: prepare a JSONL bundle for Colab inference, ingest results back."""

import json
import random
from pathlib import Path

from loguru import logger
from tqdm import tqdm

from prefvlm.config import cfg
from prefvlm.openai_client import encode_image, load_prompt
from prefvlm.runners.frontier import (
    _format_choices,
    _format_preference_profile,
    _persona_ctx,
    _pick_wrong_persona_prefs,
)


# ---------------------------------------------------------------------------
# Prepare
# ---------------------------------------------------------------------------

def prepare_batch(limit: int = 0) -> Path:
    """Build data/qwen_batch.jsonl for upload to Colab.

    Each line is a self-contained inference item:
      scenario_id, condition, persona_name, question_id, persona_id,
      model, system_text (null for baseline), user_text, image_b64
    """
    with open(cfg.paths.data_dir / "scenarios.json") as f:
        all_scenarios = json.load(f)
    with open(cfg.paths.data_dir / "questions.json") as f:
        questions = {q["question_id"]: q for q in json.load(f)}
    with open(cfg.paths.data_dir / "personas.json") as f:
        personas = {p["persona_id"]: p for p in json.load(f)}

    scenarios = all_scenarios[:limit] if limit else all_scenarios
    model = cfg.models.tested_open
    conditions = ["baseline", "oracle", "wrong_persona"]

    baseline_tmpl = load_prompt("baseline_user")
    oracle_sys_tmpl = load_prompt("oracle_system")
    oracle_user_tmpl = load_prompt("oracle_user")

    out_path = cfg.paths.data_dir / "qwen_batch.jsonl"
    skipped = 0
    written = 0

    with open(out_path, "w") as fout:
        for scenario in tqdm(scenarios, desc="Building Qwen batch"):
            sid = scenario["scenario_id"]
            question = questions[scenario["question_id"]]
            persona = personas[scenario["persona_id"]]

            # Skip if all three response files already exist
            resp_dir = cfg.paths.responses_dir / "qwen"
            if all(
                (resp_dir / f"{sid}_{c}.json").exists()
                for c in conditions
            ):
                logger.debug(f"Skipping {sid} — all responses present")
                skipped += 1
                continue

            # Skip scenarios without a preference file — oracle would be meaningless
            pref_path = cfg.paths.preferences_dir / f"{sid}.json"
            if not pref_path.exists():
                logger.debug(f"Skipping {sid} — no preference file")
                continue
            with open(pref_path) as f:
                own_prefs = json.load(f).get("preferences", [])

            rubric_path = cfg.paths.rubrics_dir / f"{sid}.json"
            own_rubric = None
            if rubric_path.exists():
                with open(rubric_path) as f:
                    own_rubric = json.load(f)

            image_path = Path(question["image_path"])
            image_b64 = encode_image(image_path)
            choices_str = _format_choices(question["choices"])

            for condition in conditions:
                # Skip if this specific response already exists
                if (resp_dir / f"{sid}_{condition}.json").exists():
                    continue

                if condition == "wrong_persona":
                    wp, wp_prefs = _pick_wrong_persona_prefs(
                        scenario, all_scenarios,
                        cfg.paths.preferences_dir, personas, cfg.seed,
                    )
                    inf_persona = wp
                    inf_prefs = wp_prefs
                    inf_rubric = own_rubric
                else:
                    inf_persona = persona
                    inf_prefs = own_prefs
                    inf_rubric = own_rubric

                # Build system and user text
                if condition == "baseline":
                    system_text = None
                    user_text = baseline_tmpl.format(
                        question=question["question"],
                        choices=choices_str,
                    )
                else:
                    ctx = _persona_ctx(inf_persona)
                    pref_str = _format_preference_profile(inf_prefs, rubric=inf_rubric)
                    system_text = oracle_sys_tmpl.format(
                        preference_profile=pref_str,
                        **ctx,
                    )
                    user_text = oracle_user_tmpl.format(
                        question=question["question"],
                        choices=choices_str,
                    )

                item = {
                    "scenario_id": sid,
                    "question_id": scenario["question_id"],
                    "persona_id": scenario["persona_id"],
                    "persona_name": inf_persona["name"],
                    "condition": condition,
                    "model": model,
                    "system_text": system_text,
                    "user_text": user_text,
                    "image_b64": image_b64,
                }
                fout.write(json.dumps(item) + "\n")
                written += 1

    size_mb = out_path.stat().st_size / 1e6
    logger.success(
        f"Batch ready: {written} items written, {skipped} scenarios skipped "
        f"→ {out_path} ({size_mb:.1f} MB)"
    )
    return out_path


# ---------------------------------------------------------------------------
# Ingest
# ---------------------------------------------------------------------------

def ingest_batch(results_path: str | None = None) -> list[dict]:
    """Ingest Colab output (data/qwen_results.jsonl) into data/responses/qwen/.

    Each line of results JSONL must have:
      scenario_id, condition, persona_name, question_id, persona_id, model, response
    """
    results_file = Path(results_path) if results_path else cfg.paths.data_dir / "qwen_results.jsonl"
    if not results_file.exists():
        raise FileNotFoundError(
            f"Results file not found: {results_file}\n"
            "Run the Colab notebook and download qwen_results.jsonl to data/"
        )

    out_dir = cfg.paths.responses_dir / "qwen"
    out_dir.mkdir(parents=True, exist_ok=True)

    ingested = []
    with open(results_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            sid = item["scenario_id"]
            condition = item["condition"]

            out_path = out_dir / f"{sid}_{condition}.json"
            result = {
                "scenario_id": sid,
                "question_id": item.get("question_id"),
                "persona_id": item.get("persona_id"),
                "condition": condition,
                "model": item.get("model", cfg.models.tested_open),
                "persona_name": item.get("persona_name", ""),
                "response": item["response"],
            }
            with open(out_path, "w") as f:
                json.dump(result, f, indent=2)
            ingested.append(result)
            logger.debug(f"Ingested {out_path.name}")

    logger.success(f"Ingested {len(ingested)} Qwen responses → {out_dir}")
    return ingested
