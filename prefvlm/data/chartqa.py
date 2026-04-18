"""Load and sample charts from ChartQA for explanation-type questions."""

import json
import random
import re
from pathlib import Path
from typing import Any

from loguru import logger
from tqdm import tqdm

from prefvlm.config import cfg

# Filter philosophy: personalization applies to the *explanation* of the answer, not
# just the answer itself. Keep almost everything. Only drop questions whose answers
# are a single token/color/yes-no with zero explanatory surface area — those give
# the model nothing to personalize on even when asked to explain its reasoning.
#
# Specifically drop only:
#   1. Color / visual-element-only questions ("what color is the tallest bar?")
#   2. Yes/No questions (answer is literally "yes" or "no")
#   3. Questions where the reference answer is a single bare number or color word

_COLOR_WORDS = {
    "red", "blue", "green", "yellow", "orange", "purple", "pink", "brown",
    "black", "white", "grey", "gray", "cyan", "magenta", "teal", "maroon",
    "navy", "olive", "lime", "indigo", "violet", "gold", "silver",
}

_VISUAL_ONLY_RE = re.compile(
    r"(what|which)\s+(color|colour|colours|colors)\s+(is|are|does|represents?|used|shown)",
    re.I,
)


def _is_explanation_question(question: str, answer: str = "") -> bool:
    """Drop only questions whose answer has no explanatory surface area at all."""
    q = question.strip()
    a = answer.strip().lower()

    # Drop color / visual-element-only questions
    if _VISUAL_ONLY_RE.search(q):
        return False

    # Drop yes/no answers
    if a in ("yes", "no", "yes.", "no."):
        return False

    # Drop if answer is a single bare color word
    if a in _COLOR_WORDS:
        return False

    # Drop if the question itself is only about the visual color of a chart element
    if re.search(r"(color|colour) of (the )?(bar|line|segment|slice|wedge|dot|point)", q, re.I):
        return False

    return True


def _infer_chart_type(question: str, answer: str) -> str:
    """Rough heuristic for chart type from question text."""
    q = (question + " " + answer).lower()
    if any(w in q for w in ["bar", "bars", "bar chart"]):
        return "bar"
    if any(w in q for w in ["line", "lines", "line chart", "time series"]):
        return "line"
    if any(w in q for w in ["pie", "donut", "proportion", "percentage breakdown"]):
        return "pie"
    if any(w in q for w in ["scatter", "correlation", "plot"]):
        return "scatter"
    if any(w in q for w in ["table", "row", "column", "cell"]):
        return "table"
    return "unknown"


def _infer_topic(question: str) -> str:
    """Rough heuristic for domain topic from question text."""
    q = question.lower()
    if any(w in q for w in ["gdp", "economic", "economy", "trade", "export", "import", "revenue", "profit", "market"]):
        return "economics"
    if any(w in q for w in ["population", "demographic", "age", "gender", "birth", "death", "migration"]):
        return "demographics"
    if any(w in q for w in ["temperature", "climate", "weather", "rainfall", "emission", "energy", "carbon"]):
        return "environment"
    if any(w in q for w in ["health", "disease", "patient", "mortality", "hospital", "medical", "infection"]):
        return "health"
    if any(w in q for w in ["student", "education", "school", "university", "learning", "literacy", "graduation"]):
        return "education"
    if any(w in q for w in ["sale", "product", "customer", "brand", "retail", "consumer"]):
        return "business"
    if any(w in q for w in ["politic", "election", "vote", "government", "party", "approval"]):
        return "politics"
    if any(w in q for w in ["sport", "athlete", "game", "team", "score", "player"]):
        return "sports"
    if any(w in q for w in ["technology", "internet", "social media", "digital", "software", "computer"]):
        return "technology"
    return "general"


def load_charts(n: int | None = None, seed: int | None = None) -> list[dict[str, Any]]:
    """Load and sample explanation-type charts from ChartQA.

    Args:
        n: Number of charts to sample. Defaults to config value.
        seed: Random seed. Defaults to config value.

    Returns:
        List of chart dicts saved to data/charts.json.
    """
    n = n or cfg.scale.n_charts
    seed = seed if seed is not None else cfg.seed

    charts_json = cfg.paths.data_dir / "charts.json"

    # Return cached if already generated
    if charts_json.exists():
        logger.info("charts.json already exists — loading from disk")
        with open(charts_json) as f:
            return json.load(f)

    logger.info("Loading ChartQA dataset from HuggingFace...")
    from datasets import load_dataset  # lazy import — slow to import

    # Try primary source, fall back to secondary
    try:
        ds = load_dataset("HuggingFaceM4/ChartQA", split="train", trust_remote_code=True)
        logger.info(f"Loaded HuggingFaceM4/ChartQA train split: {len(ds)} examples")
    except Exception as e:
        logger.warning(f"Primary source failed ({e}), trying ahmed-masry/ChartQA")
        ds = load_dataset("ahmed-masry/ChartQA", split="train", trust_remote_code=True)
        logger.info(f"Loaded ahmed-masry/ChartQA train split: {len(ds)} examples")

    # Identify column names
    col_names = ds.column_names
    logger.debug(f"Dataset columns: {col_names}")

    # Normalise column access — different versions use different names
    def get_question(row: dict) -> str:
        for k in ("query", "question", "Query", "Question"):
            if k in row and row[k]:
                return str(row[k])
        return ""

    def get_answer(row: dict) -> str:
        for k in ("label", "answer", "Label", "Answer", "labels"):
            if k in row and row[k]:
                v = row[k]
                if isinstance(v, list):
                    return str(v[0]) if v else ""
                return str(v)
        return ""

    def get_image(row: dict):
        for k in ("image", "Image", "img"):
            if k in row and row[k] is not None:
                return row[k]
        return None

    # Filter for explanation-type questions
    logger.info("Filtering for explanation-type questions...")
    candidates = []
    for i, row in enumerate(tqdm(ds, desc="Filtering")):
        q = get_question(row)
        a = get_answer(row)
        img = get_image(row)
        if q and a and img and _is_explanation_question(q, a):
            candidates.append({"idx": i, "question": q, "answer": a, "image": img})

    logger.info(f"Found {len(candidates)} explanation-type candidates from {len(ds)} total")

    if len(candidates) < n:
        raise ValueError(
            f"Only {len(candidates)} explanation-type examples found, need {n}. "
            "Try lowering n_charts in config.yaml or relaxing the filter."
        )

    # Sample n charts with fixed seed
    rng = random.Random(seed)
    sampled = rng.sample(candidates, n)

    # Save images and build output list
    cfg.paths.charts_dir.mkdir(parents=True, exist_ok=True)
    output: list[dict[str, Any]] = []

    logger.info(f"Saving {n} chart images to {cfg.paths.charts_dir}")
    for i, item in enumerate(tqdm(sampled, desc="Saving images")):
        chart_id = f"chart_{i:03d}"
        image_path = cfg.paths.charts_dir / f"{chart_id}.png"

        # PIL Image object from HuggingFace datasets
        pil_img = item["image"]
        if hasattr(pil_img, "save"):
            pil_img.save(image_path, format="PNG")
        else:
            # Bytes fallback
            image_path.write_bytes(pil_img)

        rel_path = str(image_path.relative_to(Path.cwd())) if image_path.is_relative_to(Path.cwd()) else str(image_path)

        output.append({
            "chart_id": chart_id,
            "image_path": str(image_path),
            "question": item["question"],
            "reference_answer": item["answer"],
            "chart_type": _infer_chart_type(item["question"], item["answer"]),
            "topic": _infer_topic(item["question"]),
        })

    with open(charts_json, "w") as f:
        json.dump(output, f, indent=2)
    logger.success(f"Saved {len(output)} charts to {charts_json}")

    return output
