"""Judge scoring: evaluate VLM responses against per-scenario rubrics."""

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

from loguru import logger
from tqdm import tqdm

from prefvlm.cache import cached_call
from prefvlm.config import cfg
from prefvlm.openai_client import chat_completion, load_prompt


# ---------------------------------------------------------------------------
# Rubric formatter
# ---------------------------------------------------------------------------

def _format_rubric_block(attributes: list[dict]) -> str:
    """Render rubric attributes into a readable block for the judge prompt."""
    lines = []
    for i, attr in enumerate(attributes, 1):
        lines.append(
            f"### Attribute {i}: {attr['name']}"
            f"  (type={attr.get('type','?')}, weight={attr.get('weight', 0):.3f},"
            f" preferred_value={attr.get('preferred_value', '?')})"
        )
        for lvl in attr.get("levels", []):
            lines.append(
                f"  Score {lvl['score']} — {lvl.get('label', '')}: {lvl.get('description', '')}"
            )
        lines.append("")
    return "\n".join(lines)


def _format_choices(choices: list[str]) -> str:
    labels = "ABCDEFGH"
    return "  ".join(f"{labels[i]}. {c}" for i, c in enumerate(choices))


# ---------------------------------------------------------------------------
# Single judgment
# ---------------------------------------------------------------------------

def _judge_one(
    response: dict,
    rubric: dict,
    judge_model: str,
) -> dict:
    """Score one response against its rubric. Returns judgment dict."""
    sid = response["scenario_id"]
    condition = response["condition"]
    resp_model = response.get("model", "unknown")

    # Include a short model tag in the filename so frontier and qwen don't collide
    model_tag = "qwen" if "qwen" in resp_model.lower() else "frontier"
    out_path = cfg.paths.judgments_dir / f"{sid}_{condition}_{model_tag}.json"
    if out_path.exists():
        logger.debug(f"Judgment cache hit: {out_path.name}")
        with open(out_path) as f:
            return json.load(f)

    attributes = rubric.get("attributes", [])
    if not attributes:
        raise ValueError(f"Rubric for {sid} has no attributes")

    rubric_block = _format_rubric_block(attributes)
    choices_str = _format_choices(response.get("choices", []))  # may not be present

    # Load question text from the rubric's embedded question_id
    question_text = response.get("question", rubric.get("question_id", ""))

    prompt_template = load_prompt("judge_scoring")
    prompt = prompt_template.format(
        question=question_text,
        choices=choices_str,
        response=response["response"],
        rubric_block=rubric_block,
    )

    raw = cached_call(
        "judgment",
        (sid, condition, resp_model, judge_model),
        lambda: chat_completion(
            model=judge_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=cfg.inference.temperature_judge,
            max_tokens=2000,
            json_mode=True,
        ),
    )

    try:
        parsed = json.loads(raw)
        scores_list = parsed.get("scores", parsed) if isinstance(parsed, dict) else parsed
        if not isinstance(scores_list, list):
            raise ValueError(f"Expected list under 'scores', got {type(scores_list)}")
    except Exception as e:
        logger.error(f"Judge parse error for {sid}/{condition}: {e}\nRaw: {raw[:300]}")
        raise

    # Build name → rubric attr lookup for weight
    attr_by_name = {a["name"]: a for a in attributes}

    # Compute weighted score
    total_weight = 0.0
    weighted_sum = 0.0
    scored_attrs = []
    for entry in scores_list:
        name = entry.get("name", "")
        score = float(entry.get("score", 3))
        attr = attr_by_name.get(name, {})
        weight = float(attr.get("weight", 1.0 / max(len(attributes), 1)))
        scored_attrs.append({
            "name": name,
            "score": score,
            "weight": weight,
            "rationale": entry.get("rationale", ""),
        })
        weighted_sum += score * weight
        total_weight += weight

    weighted_score = round(weighted_sum / total_weight, 4) if total_weight > 0 else 0.0

    judgment = {
        "scenario_id": sid,
        "persona_id": response.get("persona_id"),
        "question_id": response.get("question_id"),
        "condition": condition,
        "response_model": resp_model,
        "judge_model": judge_model,
        "weighted_score": weighted_score,
        "n_attributes": len(scored_attrs),
        "attribute_scores": scored_attrs,
    }

    cfg.paths.judgments_dir.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(judgment, f, indent=2)
    logger.debug(f"Saved judgment → {out_path.name}  (weighted={weighted_score:.3f})")
    return judgment


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------

def score_all(limit: int = 0) -> list[dict]:
    """Score all frontier (and qwen if present) responses.

    Args:
        limit: Process only first N scenarios (0 = all).

    Returns:
        List of judgment dicts.
    """
    judge_model = cfg.models.judge

    # Collect response files: frontier first, then qwen
    response_files: list[Path] = []
    frontier_dir = cfg.paths.responses_dir / "frontier"
    if frontier_dir.exists():
        response_files.extend(sorted(frontier_dir.glob("*.json")))
    qwen_dir = cfg.paths.responses_dir / "qwen"
    if qwen_dir.exists():
        response_files.extend(sorted(qwen_dir.glob("*.json")))

    # Load rubrics index: scenario_id → rubric
    rubrics: dict[str, dict] = {}
    rubrics_dir = cfg.paths.rubrics_dir
    if rubrics_dir.exists():
        for rf in rubrics_dir.glob("*.json"):
            with open(rf) as f:
                r = json.load(f)
            rubrics[r["scenario_id"]] = r

    # Load questions for question text
    questions: dict[str, dict] = {}
    qfile = cfg.paths.data_dir / "questions.json"
    if qfile.exists():
        with open(qfile) as f:
            for q in json.load(f):
                questions[q["question_id"]] = q

    # Optionally limit by unique scenario_ids seen
    if limit:
        seen_sids: list[str] = []
        filtered: list[Path] = []
        for rf in response_files:
            # Extract scenario_id from filename (e.g. s0001_baseline.json → s0001)
            sid = rf.stem.split("_", 1)[0]  # s0001_wrong_persona → s0001
            if sid not in seen_sids:
                seen_sids.append(sid)
            if seen_sids.index(sid) < limit:
                filtered.append(rf)
        response_files = filtered

    def _judge_file(rf: Path) -> dict | None:
        with open(rf) as f:
            response = json.load(f)
        sid = response["scenario_id"]
        qid = response.get("question_id")
        if "question" not in response and qid and qid in questions:
            q = questions[qid]
            response["question"] = q["question"]
            response["choices"] = q.get("choices", [])
        rubric = rubrics.get(sid)
        if rubric is None:
            logger.warning(f"No rubric for {sid}, skipping {rf.name}")
            return None
        try:
            judgment = _judge_one(response, rubric, judge_model)
            logger.info(
                f"[{sid}] {response['condition']:12s} → "
                f"weighted={judgment['weighted_score']:.3f}  "
                f"({judgment['n_attributes']} attrs)"
            )
            return judgment
        except Exception as e:
            logger.error(f"[{sid}] {response.get('condition')} judge FAILED: {e}")
            return None

    results = []
    with tqdm(total=len(response_files), desc="Judging responses") as pbar:
        with ThreadPoolExecutor(max_workers=20) as pool:
            futures = {pool.submit(_judge_file, rf): rf for rf in response_files}
            for future in as_completed(futures):
                try:
                    j = future.result()
                    if j:
                        results.append(j)
                except Exception as e:
                    logger.error(f"Judge worker error: {e}")
                pbar.update(1)

    logger.success(f"Judging complete: {len(results)} judgments")
    return results


def summarize(judgments: Optional[list[dict]] = None) -> dict:
    """Load all saved judgments and compute per-condition mean weighted scores."""
    if judgments is None:
        judgments = []
        if cfg.paths.judgments_dir.exists():
            for jf in sorted(cfg.paths.judgments_dir.glob("*.json")):
                with open(jf) as f:
                    judgments.append(json.load(f))

    from collections import defaultdict
    by_condition: dict[str, list[float]] = defaultdict(list)
    for j in judgments:
        by_condition[j["condition"]].append(j["weighted_score"])

    summary = {}
    for cond, scores in by_condition.items():
        summary[cond] = {
            "n": len(scores),
            "mean": round(sum(scores) / len(scores), 4),
            "min": round(min(scores), 4),
            "max": round(max(scores), 4),
        }
    return summary
