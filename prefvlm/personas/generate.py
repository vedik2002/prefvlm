"""Generate diverse personas with rejection sampling for near-duplicates."""

import json
import math
from typing import Optional

from loguru import logger
from tqdm import tqdm

from prefvlm.cache import cached_call
from prefvlm.config import cfg
from prefvlm.openai_client import chat_completion, get_embedding, load_prompt
from prefvlm.personas.schema import Persona

# Stratification slots: (data_literacy, age_range, occupation_type, location_constraint, name_constraint)
# Geography spread: urban / suburban / small-town / rural / international
# Name constraint ensures ethnic/surname diversity — no repeated surnames across slots
_SLOTS = [
    (
        "low",
        "24–35",
        "occupation with genuinely little data contact: e.g. restaurant cook, warehouse worker, home health aide, landscaper, postal worker, early childhood caregiver, retail associate. Do NOT choose a trades job that involves energy reports or safety data tracking.",
        "small town or rural US location (e.g. a town in Mississippi, Appalachia, rural Midwest, or similar)",
        "Latino/Hispanic or Southeast Asian name — first and last name both",
    ),
    (
        "moderate",
        "38–52",
        "occupation with occasional data contact but not a primary analyst role: e.g. classroom teacher, nurse, journalist, small business owner, social worker, middle manager at a retail chain",
        "mid-size US city (e.g. Cleveland, Albuquerque, Louisville, Des Moines, Baton Rouge — not NYC, LA, SF, Seattle, or Portland)",
        "African American or West African name — first and last name both",
    ),
    (
        "moderate",
        "55–70",
        "occupation with occasional data contact: e.g. nurse, school principal, small business owner, sales manager, local government administrator. The person should have been in their career for 20+ years.",
        "suburban or regional US location (e.g. a suburb of a mid-size city, or a smaller city in the South or Midwest)",
        "Anglo or Eastern European name, but a DIFFERENT surname from all other personas",
    ),
    (
        "high",
        "28–40",
        "occupation where data analysis is a PRIMARY job function: e.g. data analyst, software engineer, financial analyst, product manager, epidemiologist, quantitative researcher, business intelligence developer",
        "major international city outside the US (e.g. Toronto, London, Berlin, Singapore, São Paulo, Lagos, Mumbai, Sydney)",
        "East Asian or South Asian name — first and last name both",
    ),
    (
        "high",
        "44–58",
        "occupation where data analysis is a PRIMARY job function: e.g. senior data scientist, research director, operations research analyst, chief financial officer, biostatistician, policy analyst at a think tank",
        "large US metro area (e.g. Chicago, Houston, Boston, Atlanta, Miami, Denver — not the same city as any other persona)",
        "Middle Eastern, North African, or Eastern European name — first and last name both",
    ),
]


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _generate_one(
    persona_id: str,
    data_literacy: str,
    age_range: str,
    occupation_type: str,
    location_constraint: str,
    name_constraint: str,
) -> Optional[Persona]:
    """Call GPT to generate a single persona. Returns None on parse failure."""
    prompt_template = load_prompt("persona_generation")
    prompt = prompt_template.format(
        persona_id=persona_id,
        data_literacy=data_literacy,
        age_range=age_range,
        occupation_type=occupation_type,
        location_constraint=location_constraint,
        name_constraint=name_constraint,
    )

    def _call() -> str:
        return chat_completion(
            model=cfg.models.persona_gen,
            messages=[{"role": "user", "content": prompt}],
            temperature=cfg.inference.temperature_generation,
            max_tokens=600,
            json_mode=True,
        )

    # Don't cache individual generation calls — we want variety on retries
    raw = _call()

    try:
        data = json.loads(raw)
        # Ensure persona_id is set correctly
        data["persona_id"] = persona_id
        return Persona.model_validate(data)
    except Exception as e:
        logger.warning(f"Failed to parse persona {persona_id}: {e}\nRaw: {raw[:300]}")
        return None


def generate_personas(n: int | None = None) -> list[Persona]:
    """Generate n personas with stratification and duplicate rejection.

    Args:
        n: Number of personas. Defaults to config value (5).

    Returns:
        List of validated Persona objects, also saved to data/personas.json.
    """
    n = n or cfg.scale.n_personas
    personas_path = cfg.paths.data_dir / "personas.json"

    if personas_path.exists():
        logger.info("personas.json already exists — loading from disk")
        with open(personas_path) as f:
            raw = json.load(f)
        return [Persona.model_validate(p) for p in raw]

    if n > len(_SLOTS):
        raise ValueError(
            f"n_personas={n} exceeds the number of stratification slots ({len(_SLOTS)}). "
            "Add more slots to _SLOTS in generate.py."
        )

    personas: list[Persona] = []
    embeddings: list[list[float]] = []
    similarity_threshold = 0.85
    max_retries = 3

    logger.info(f"Generating {n} personas (model={cfg.models.persona_gen})...")

    for slot_idx in tqdm(range(n), desc="Generating personas"):
        data_literacy, age_range, occupation_type, location_constraint, name_constraint = _SLOTS[slot_idx]
        persona_id = f"p{slot_idx + 1:03d}"

        accepted: Optional[Persona] = None
        for attempt in range(1, max_retries + 1):
            logger.debug(
                f"Slot {slot_idx + 1}/{n} ({data_literacy}, {age_range}) — attempt {attempt}"
            )
            candidate = _generate_one(persona_id, data_literacy, age_range, occupation_type, location_constraint, name_constraint)
            if candidate is None:
                logger.warning(f"  Parse failed on attempt {attempt}, retrying...")
                continue

            # Rejection sampling: check cosine similarity against all accepted backstories
            if embeddings:
                cand_emb = get_embedding(candidate.backstory)
                max_sim = max(_cosine_similarity(cand_emb, e) for e in embeddings)
                if max_sim > similarity_threshold:
                    logger.warning(
                        f"  Candidate too similar to existing persona "
                        f"(sim={max_sim:.3f} > {similarity_threshold}), retrying..."
                    )
                    continue
                embeddings.append(cand_emb)
            else:
                embeddings.append(get_embedding(candidate.backstory))

            accepted = candidate
            break

        if accepted is None:
            raise RuntimeError(
                f"Failed to generate a valid, non-duplicate persona for slot {slot_idx + 1} "
                f"after {max_retries} attempts."
            )

        personas.append(accepted)
        logger.info(
            f"  ✓ {accepted.name} ({accepted.age}, {accepted.occupation}) "
            f"data_literacy={accepted.data_literacy}"
        )

    # Save
    cfg.paths.data_dir.mkdir(parents=True, exist_ok=True)
    with open(personas_path, "w") as f:
        json.dump([p.model_dump() for p in personas], f, indent=2)
    logger.success(f"Saved {len(personas)} personas to {personas_path}")

    return personas
