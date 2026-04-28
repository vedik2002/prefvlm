"""Generate diverse personas with enforced Big Five variation and domain deduplication."""

import json
import math
import random
import re
from typing import Optional

from loguru import logger
from tqdm import tqdm

from prefvlm.cache import cached_call
from prefvlm.config import cfg
from prefvlm.openai_client import chat_completion, get_embedding, load_prompt
from prefvlm.personas.schema import Persona

# ---------------------------------------------------------------------------
# Diversity pools
# ---------------------------------------------------------------------------

_AGE_RANGES = [
    "18–25", "22–30", "26–35", "30–42",
    "38–50", "45–55", "52–65", "60–72",
]

_OCCUPATION_POOL = [
    "line cook or sous chef in a restaurant kitchen",
    "registered nurse in a hospital or clinic",
    "elementary school teacher",
    "software engineer at a mid-size tech company",
    "farmer or agricultural worker",
    "long-haul truck driver",
    "licensed electrician or plumber",
    "financial analyst or accountant",
    "social worker or case manager",
    "auto mechanic or diesel technician",
    "high school science or math teacher",
    "physical therapist or occupational therapist",
    "journalist or copy editor at a regional newspaper",
    "real estate agent or property manager",
    "construction project supervisor",
    "librarian or archivist",
    "marine biologist or field ecologist",
    "retail store manager",
    "EMT or paramedic",
    "civil or structural engineer",
    "yoga instructor or personal fitness trainer",
    "dentist or dental hygienist",
    "event coordinator or wedding planner",
    "master carpenter or furniture maker",
    "data analyst at a healthcare company",
    "veterinarian or vet tech",
    "barista or independent café owner",
    "recording studio musician or audio engineer",
    "park ranger or wildlife technician",
    "HR manager or recruiting coordinator",
    "geologist or hydrogeologist",
    "flight attendant or airline operations agent",
    "urban planner at a city government",
    "commercial fisherman or boat captain",
    "pastry chef or bakery owner",
    "documentary filmmaker or video editor",
    "biomedical or electrical engineer",
    "speech-language pathologist",
    "rideshare driver with a side gig",
    "museum curator or exhibit designer",
    "food scientist or quality-control specialist",
    "sign language interpreter",
    "sommelier or beverage consultant",
    "police officer or sheriff's deputy",
    "insurance adjuster or claims specialist",
    "tour guide or travel blogger",
    "pharmacist at a community pharmacy",
    "oil-rig or pipeline field technician",
    "childcare director or preschool teacher",
    "hospital administrator or health services manager",
]

_LOCATION_POOL = [
    "a small town in rural Mississippi",
    "a suburb of Cincinnati, Ohio",
    "Chicago, Illinois",
    "a small ranching town in rural Montana",
    "Houston, Texas",
    "Portland, Oregon",
    "Nashville, Tennessee",
    "a rural community in the Appalachian mountains",
    "Phoenix, Arizona",
    "Detroit, Michigan",
    "New Orleans, Louisiana",
    "Salt Lake City, Utah",
    "Boston, Massachusetts",
    "a small farming town in rural Iowa",
    "Denver, Colorado",
    "Miami, Florida",
    "Pittsburgh, Pennsylvania",
    "rural eastern Washington State",
    "Minneapolis, Minnesota",
    "Las Vegas, Nevada",
    "a coastal town in rural Maine",
    "San Diego, California",
    "Memphis, Tennessee",
    "Indianapolis, Indiana",
    "Albuquerque, New Mexico",
    "a small town in the Texas Hill Country",
    "Baltimore, Maryland",
    "a rural community in rural Louisiana bayou country",
    "Kansas City, Missouri",
    "Tampa, Florida",
    "Cleveland, Ohio",
    "Sacramento, California",
    "a small city in rural Vermont",
    "Columbus, Ohio",
    "Louisville, Kentucky",
    "Raleigh, North Carolina",
    "Boise, Idaho",
    "rural eastern Oregon",
    "Omaha, Nebraska",
    "Tulsa, Oklahoma",
    "Toronto, Canada",
    "London, UK",
    "Berlin, Germany",
    "Singapore",
    "Mumbai, India",
    "Lagos, Nigeria",
    "Sydney, Australia",
    "São Paulo, Brazil",
    "Seoul, South Korea",
    "Mexico City, Mexico",
]

_NAME_POOLS = [
    "Latino or Hispanic first and last name",
    "West African or Nigerian first and last name",
    "East Asian (Chinese, Japanese, or Korean) first and last name",
    "South Asian (Indian or Pakistani) first and last name",
    "Anglo or Irish-American first and last name",
    "Eastern European (Polish, Ukrainian, or Czech) first and last name",
    "Middle Eastern or North African first and last name",
    "Scandinavian first and last name",
    "African American first and last name",
    "Southeast Asian (Filipino, Vietnamese, or Thai) first and last name",
]

_EDUCATION_POOL = [
    "high school diploma",
    "vocational or trade school certificate",
    "some college, no degree",
    "associate degree",
    "bachelor's degree",
    "bachelor's degree",   # weighted slightly more common
    "master's degree",
    "doctoral degree",
    "professional degree (MD, JD, or equivalent)",
]

# Big Five levels and their display strings
_BF_LEVELS = ["low", "moderate", "high"]
_BF_TRAITS = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]


def _generate_big_five_profiles(n: int, seed: int) -> list[dict]:
    """Generate n Big Five profiles ensuring each trait has balanced level coverage.

    Constraint: no single level appears in more than ceil(n * 0.5) personas per trait.
    """
    rng = random.Random(seed)
    counts = {t: {l: 0 for l in _BF_LEVELS} for t in _BF_TRAITS}
    profiles = []

    for _ in range(n):
        profile = {}
        for trait in _BF_TRAITS:
            # Prefer under-represented levels; break ties randomly
            min_count = min(counts[trait].values())
            candidates = [l for l in _BF_LEVELS if counts[trait][l] == min_count]
            # Add moderate as a small extra candidate for realism
            if "moderate" not in candidates and rng.random() < 0.15:
                candidates.append("moderate")
            level = rng.choice(candidates)
            profile[trait] = level
            counts[trait][level] += 1
        profiles.append(profile)

    rng.shuffle(profiles)

    # Log trait distribution
    for trait in _BF_TRAITS:
        dist = {l: counts[trait][l] for l in _BF_LEVELS}
        logger.debug(f"Big Five {trait}: {dist}")

    return profiles


def _build_slots(n: int, seed: int) -> list[dict]:
    """Build n persona slot specs from diversity pools."""
    rng = random.Random(seed)
    big_five_profiles = _generate_big_five_profiles(n, seed)

    # Shuffle pools and cycle through them
    ages = (_AGE_RANGES * math.ceil(n / len(_AGE_RANGES)))[:n]
    occupations = rng.sample(_OCCUPATION_POOL, min(n, len(_OCCUPATION_POOL)))
    if n > len(_OCCUPATION_POOL):
        occupations += rng.choices(_OCCUPATION_POOL, k=n - len(_OCCUPATION_POOL))
    locations = rng.sample(_LOCATION_POOL, min(n, len(_LOCATION_POOL)))
    if n > len(_LOCATION_POOL):
        locations += rng.choices(_LOCATION_POOL, k=n - len(_LOCATION_POOL))
    names = (_NAME_POOLS * math.ceil(n / len(_NAME_POOLS)))
    rng.shuffle(names)
    names = names[:n]

    rng.shuffle(ages)

    slots = []
    for i in range(n):
        slots.append({
            "age_range": ages[i],
            "occupation_hint": occupations[i],
            "location_constraint": locations[i],
            "name_constraint": names[i],
            "big_five": big_five_profiles[i],
        })
    return slots


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _extract_domain_keywords(domains: list[str]) -> set[str]:
    stop = {"and", "or", "the", "of", "in", "for", "a", "an", "to", "at", "by", "with"}
    words = set()
    for d in domains:
        for w in re.split(r"\W+", d.lower()):
            if w and w not in stop and len(w) > 2:
                words.add(w)
    return words


def _format_big_five(bf: dict) -> str:
    return (
        f"openness={bf['openness']}, conscientiousness={bf['conscientiousness']}, "
        f"extraversion={bf['extraversion']}, agreeableness={bf['agreeableness']}, "
        f"neuroticism={bf['neuroticism']}"
    )


# ---------------------------------------------------------------------------
# Core generation
# ---------------------------------------------------------------------------

def _generate_one(
    persona_id: str,
    slot: dict,
    forbidden_domain_keywords: set[str],
) -> Optional[Persona]:
    """Generate one persona from a slot spec. Returns None on parse failure."""
    prompt_template = load_prompt("persona_generation")
    forbidden_str = (
        ", ".join(sorted(forbidden_domain_keywords)) if forbidden_domain_keywords else "none"
    )
    prompt = prompt_template.format(
        persona_id=persona_id,
        age_range=slot["age_range"],
        occupation_hint=slot["occupation_hint"],
        location_constraint=slot["location_constraint"],
        name_constraint=slot["name_constraint"],
        big_five_constraints=_format_big_five(slot["big_five"]),
        forbidden_domain_keywords=forbidden_str,
    )

    raw = chat_completion(
        model=cfg.models.persona_gen,
        messages=[{"role": "user", "content": prompt}],
        temperature=cfg.inference.temperature_generation,
        max_tokens=700,
        json_mode=True,
    )

    try:
        data = json.loads(raw)
        data["persona_id"] = persona_id
        data["big_five"] = slot["big_five"]   # inject — model cannot deviate
        return Persona.model_validate(data)
    except Exception as e:
        logger.warning(f"Failed to parse persona {persona_id}: {e}\nRaw: {raw[:300]}")
        return None


def generate_personas(n: int | None = None) -> list[Persona]:
    """Generate n personas with diverse Big Five profiles and backstory deduplication."""
    n = n or cfg.scale.n_personas
    personas_path = cfg.paths.data_dir / "personas.json"

    if personas_path.exists():
        logger.info("personas.json already exists — loading from disk")
        with open(personas_path) as f:
            raw_list = json.load(f)
        return [Persona.model_validate(p) for p in raw_list]

    slots = _build_slots(n, cfg.seed)

    personas: list[Persona] = []
    embeddings: list[list[float]] = []
    used_domain_keywords: set[str] = set()
    similarity_threshold = 0.85
    max_retries = 4

    logger.info(f"Generating {n} personas (model={cfg.models.persona_gen})...")

    for slot_idx in tqdm(range(n), desc="Generating personas"):
        slot = slots[slot_idx]
        persona_id = f"p{slot_idx + 1:03d}"

        accepted: Optional[Persona] = None
        for attempt in range(1, max_retries + 1):
            candidate = _generate_one(persona_id, slot, used_domain_keywords)
            if candidate is None:
                logger.warning(f"  p{slot_idx+1} attempt {attempt}: parse failed, retrying...")
                continue

            # Backstory cosine similarity rejection sampling
            cand_emb = get_embedding(candidate.backstory)
            if embeddings:
                max_sim = max(_cosine_similarity(cand_emb, e) for e in embeddings)
                if max_sim > similarity_threshold:
                    logger.warning(
                        f"  p{slot_idx+1} attempt {attempt}: backstory too similar "
                        f"(sim={max_sim:.3f}), retrying..."
                    )
                    continue

            embeddings.append(cand_emb)
            new_keywords = _extract_domain_keywords(candidate.domain_familiarity)
            overlap = new_keywords & used_domain_keywords
            if overlap:
                logger.debug(f"  p{slot_idx+1}: domain overlap: {overlap}")

            accepted = candidate
            break

        if accepted is None:
            raise RuntimeError(
                f"Failed to generate valid persona for slot {slot_idx + 1} "
                f"after {max_retries} attempts."
            )

        used_domain_keywords.update(_extract_domain_keywords(accepted.domain_familiarity))
        personas.append(accepted)
        logger.info(
            f"  ✓ p{slot_idx+1:02d} {accepted.name} "
            f"({accepted.age}, {accepted.occupation[:40]})"
        )

    cfg.paths.data_dir.mkdir(parents=True, exist_ok=True)
    with open(personas_path, "w") as f:
        json.dump([p.model_dump() for p in personas], f, indent=2)
    logger.success(f"Saved {len(personas)} personas → {personas_path}")
    return personas
