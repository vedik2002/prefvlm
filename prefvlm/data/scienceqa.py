"""Load, deduplicate, and sample questions from ScienceQA + MMMU for VLM explanation tasks.

ScienceQA (K-12): grades 5-12, natural science, image present, solution >= 2 sentences.
MMMU (college-level): Biology, Chemistry, Physics subjects, single-image questions.

After loading both sources, deduplicate by question template fingerprint (max 2 per template),
then stratify-sample across topics.
"""

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import random

from loguru import logger
from tqdm import tqdm

from prefvlm.config import cfg

_SQA_TOPICS = ("physics", "chemistry", "biology", "earth science")

_SQA_TOPIC_MAP = {
    "physics": "physics",
    "physical-science": "physics",
    "physical science": "physics",
    "chemistry": "chemistry",
    "biology": "biology",
    "life-science": "biology",
    "life science": "biology",
    "earth-science": "earth science",
    "earth science": "earth science",
    "earth and space science": "earth science",
    "astronomy": "earth science",
    "environmental-science": "earth science",
    "environmental science": "earth science",
}

_MMMU_SUBJECT_TO_TOPIC = {
    "Biology": "biology",
    "Chemistry": "chemistry",
    "Physics": "physics",
}

# Stop words for fingerprint
_STOP = {
    "a", "an", "the", "is", "are", "was", "were", "of", "in", "on", "at",
    "to", "for", "and", "or", "not", "this", "that", "with", "by", "it",
    "its", "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "what", "which", "how", "when", "where", "who", "why", "if", "as",
    "from", "about", "into", "through", "during", "following", "between",
}


def _question_fingerprint(question: str) -> str:
    """Template fingerprint: first 6 non-stop content words, lowercased, numbers→#.

    Two questions with the same fingerprint are likely from the same template
    (e.g. 'which solution has higher concentration of X particles').
    """
    s = re.sub(r"\d+(\.\d+)?", "#", question.lower())
    s = re.sub(r"[^\w\s#]", " ", s)
    words = [w for w in s.split() if w not in _STOP and len(w) > 1]
    return " ".join(words[:6])


def _solution_sentence_count(solution: str) -> int:
    if not solution:
        return 0
    sentences = re.split(r"[.!?]+(?:\s+|$)", solution.strip())
    return len([s for s in sentences if s.strip()])


def _parse_grade(grade_val: Any) -> int | None:
    if isinstance(grade_val, int):
        return grade_val
    if isinstance(grade_val, str):
        m = re.search(r"\d+", grade_val)
        return int(m.group()) if m else None
    return None


def _format_choices(choices: list[str]) -> str:
    labels = "ABCDEFGH"
    return "  ".join(f"{labels[i]}. {c}" for i, c in enumerate(choices))


# --------------------------------------------------------------------------- #
# ScienceQA loader                                                             #
# --------------------------------------------------------------------------- #

def _load_scienceqa_candidates() -> list[dict]:
    """Return all qualifying ScienceQA candidates with topic, image, etc."""
    logger.info("Loading ScienceQA from HuggingFace (derek-thomas/ScienceQA)…")
    from datasets import load_dataset

    try:
        ds = load_dataset("derek-thomas/ScienceQA", split="train", trust_remote_code=False)
    except Exception as e:
        logger.warning(f"Primary SQA source failed ({e}), trying allenai/scienceqa")
        ds = load_dataset("allenai/scienceqa", split="train", trust_remote_code=False)

    logger.info(f"Loaded ScienceQA train: {len(ds)} examples  columns: {ds.column_names}")

    def get_image(row): return row.get("image") or row.get("Image")
    def get_subject(row): return str(row.get("subject", "")).lower().strip()
    def get_topic(row): return str(row.get("topic", "")).strip()
    def get_grade(row): return row.get("grade") or row.get("grade_level")
    def get_solution(row): return str(row.get("solution") or "").strip()
    def get_choices(row): return list(row.get("choices") or [])
    def get_answer(row):
        a = row.get("answer") or row.get("answer_idx") or 0
        return int(a)
    def get_lecture(row): return str(row.get("lecture") or "").strip()
    def get_hint(row): return str(row.get("hint") or "").strip()

    candidates = []
    for row in tqdm(ds, desc="Filtering ScienceQA"):
        if get_image(row) is None:
            continue
        if "natural" not in get_subject(row):
            continue
        grade = _parse_grade(get_grade(row))
        if grade is None or grade < 5:
            continue
        solution = get_solution(row)
        if _solution_sentence_count(solution) < 2:
            continue
        topic_raw = get_topic(row)
        topic = _SQA_TOPIC_MAP.get(topic_raw.lower())
        if topic is None:
            continue
        choices = get_choices(row)
        if len(choices) < 2:
            continue

        candidates.append({
            "source": "scienceqa",
            "question": str(row.get("question", "")),
            "choices": choices,
            "answer_index": get_answer(row),
            "lecture": get_lecture(row),
            "solution": solution,
            "subject": get_subject(row),
            "topic": topic,
            "grade": grade,
            "image": get_image(row),
            "hint": get_hint(row),
            "level": "k12",
        })

    logger.info(f"ScienceQA candidates: {len(candidates)}")
    return candidates


# --------------------------------------------------------------------------- #
# MMMU loader (college-level)                                                  #
# --------------------------------------------------------------------------- #

def _load_mmmu_candidates() -> list[dict]:
    """Return college-level science candidates from MMMU (single-image only)."""
    from datasets import load_dataset

    all_candidates = []
    for mmmu_subject, topic in _MMMU_SUBJECT_TO_TOPIC.items():
        try:
            ds = load_dataset("MMMU/MMMU", mmmu_subject, split="validation",
                              trust_remote_code=False)
            logger.info(f"MMMU {mmmu_subject}: {len(ds)} examples  columns: {ds.column_names}")
        except Exception as e:
            logger.warning(f"MMMU {mmmu_subject} failed: {e}")
            continue

        for row in ds:
            # Only multiple-choice questions
            if row.get("question_type", "") != "multiple-choice":
                continue
            # Only single-image questions
            img = row.get("image_1")
            if img is None:
                continue
            if row.get("image_2") is not None:
                continue

            # Parse options — MMMU stores as Python-list-string or actual list
            options_raw = row.get("options", [])
            if isinstance(options_raw, str):
                try:
                    import ast
                    options = ast.literal_eval(options_raw)
                except Exception:
                    options = [o.strip() for o in options_raw.split(",") if o.strip()]
            else:
                options = list(options_raw)

            if len(options) < 2:
                continue

            # Answer is a letter like "A", "B", "C", "D"
            answer_str = str(row.get("answer", "A")).strip().upper()
            answer_idx = ord(answer_str[0]) - ord("A") if answer_str else 0
            if answer_idx >= len(options):
                answer_idx = 0

            # Build a solution from explanation if available
            explanation = str(row.get("explanation", "") or "").strip()
            if not explanation or len(explanation.split()) < 8:
                continue  # skip if no meaningful explanation

            all_candidates.append({
                "source": "mmmu",
                "question": str(row.get("question", "")),
                "choices": options,
                "answer_index": answer_idx,
                "lecture": "",
                "solution": explanation,
                "subject": mmmu_subject.lower(),
                "topic": topic,
                "grade": 13,          # college-level marker
                "image": img,
                "hint": "",
                "level": "college",
            })

    logger.info(f"MMMU college-level candidates: {len(all_candidates)}")
    return all_candidates


# --------------------------------------------------------------------------- #
# Deduplication                                                                #
# --------------------------------------------------------------------------- #

def _deduplicate(candidates: list[dict], max_per_template: int = 2) -> list[dict]:
    """Keep at most max_per_template questions per fingerprint cluster."""
    by_fingerprint: dict[str, list[dict]] = defaultdict(list)
    for c in candidates:
        fp = _question_fingerprint(c["question"])
        by_fingerprint[fp].append(c)

    logger.info(f"Dedup: {len(candidates)} candidates → {len(by_fingerprint)} distinct templates")
    kept = []
    for fp, group in by_fingerprint.items():
        kept.extend(group[:max_per_template])

    logger.info(f"Dedup: kept {len(kept)} after max {max_per_template} per template")
    return kept


# --------------------------------------------------------------------------- #
# Main entry point                                                             #
# --------------------------------------------------------------------------- #

def load_questions(n: int | None = None, seed: int | None = None) -> list[dict[str, Any]]:
    """Load, deduplicate, and stratify-sample science questions.

    Blends ScienceQA (K-12) and MMMU (college-level). Returns list saved to
    data/questions.json. n is approximate — actual count may be slightly lower
    after deduplication.
    """
    n = n or cfg.scale.n_questions
    seed = seed if seed is not None else cfg.seed

    questions_json = cfg.paths.data_dir / "questions.json"
    if questions_json.exists():
        logger.info("questions.json already exists — loading from disk")
        with open(questions_json) as f:
            return json.load(f)

    # Load both sources
    sqa_candidates = _load_scienceqa_candidates()
    mmmu_candidates = _load_mmmu_candidates()

    # Separate dedup for each source
    sqa_deduped = _deduplicate(sqa_candidates, max_per_template=2)
    mmmu_deduped = _deduplicate(mmmu_candidates, max_per_template=2)

    # Budget: ~60% ScienceQA, ~40% MMMU (college), minimum 5 MMMU if available
    n_mmmu = min(max(int(n * 0.35), 5), len(mmmu_deduped))
    n_sqa = n - n_mmmu

    rng = random.Random(seed)

    # Stratified sample from ScienceQA across topics
    sqa_by_topic: dict[str, list[dict]] = defaultdict(list)
    for c in sqa_deduped:
        sqa_by_topic[c["topic"]].append(c)

    n_per_sqa_topic = n_sqa // len(_SQA_TOPICS)
    sqa_remainder = n_sqa % len(_SQA_TOPICS)
    sqa_sampled: list[dict] = []
    for i, topic in enumerate(_SQA_TOPICS):
        bucket = sqa_by_topic.get(topic, [])
        want = n_per_sqa_topic + (1 if i < sqa_remainder else 0)
        want = min(want, len(bucket))
        sqa_sampled.extend(rng.sample(bucket, want))

    # Stratified sample from MMMU across topics
    mmmu_by_topic: dict[str, list[dict]] = defaultdict(list)
    for c in mmmu_deduped:
        mmmu_by_topic[c["topic"]].append(c)

    mmmu_topics = [t for t in _MMMU_SUBJECT_TO_TOPIC.values() if mmmu_by_topic.get(t)]
    n_per_mmmu_topic = n_mmmu // max(len(mmmu_topics), 1)
    mmmu_sampled: list[dict] = []
    for topic in mmmu_topics:
        bucket = mmmu_by_topic[topic]
        want = min(n_per_mmmu_topic, len(bucket))
        mmmu_sampled.extend(rng.sample(bucket, want))
    # Top up if we're short due to rounding
    remaining_mmmu = [c for c in mmmu_deduped if c not in mmmu_sampled]
    while len(mmmu_sampled) < n_mmmu and remaining_mmmu:
        pick = rng.choice(remaining_mmmu)
        mmmu_sampled.append(pick)
        remaining_mmmu.remove(pick)

    combined = sqa_sampled + mmmu_sampled
    rng.shuffle(combined)

    logger.info(
        f"Final sample: {len(sqa_sampled)} ScienceQA K-12 + {len(mmmu_sampled)} MMMU college "
        f"= {len(combined)} total"
    )

    # Save images and build output
    cfg.paths.images_dir.mkdir(parents=True, exist_ok=True)
    output: list[dict[str, Any]] = []

    for i, item in enumerate(tqdm(combined, desc="Saving images")):
        qid = f"q{i:03d}"
        image_path = cfg.paths.images_dir / f"{qid}.png"

        pil_img = item["image"]
        if hasattr(pil_img, "save"):
            from PIL import Image
            if getattr(pil_img, "mode", "RGB") in ("RGBA", "P"):
                pil_img = pil_img.convert("RGB")
            pil_img.save(image_path, format="PNG")
        else:
            image_path.write_bytes(pil_img)

        output.append({
            "question_id": qid,
            "image_path": str(image_path),
            "question": item["question"],
            "choices": item["choices"],
            "answer_index": item["answer_index"],
            "lecture": item["lecture"],
            "solution": item["solution"],
            "subject": item["subject"],
            "topic": item["topic"],
            "grade": item["grade"],
            "hint": item.get("hint", ""),
            "level": item.get("level", "k12"),
            "source": item.get("source", "scienceqa"),
        })

    with open(questions_json, "w") as f:
        json.dump(output, f, indent=2)
    logger.success(f"Saved {len(output)} questions to {questions_json}")
    return output
