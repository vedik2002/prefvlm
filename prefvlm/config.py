"""Load and expose project-wide configuration from config.yaml and .env."""

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

load_dotenv()

_ROOT = Path(__file__).parent.parent
_CONFIG_PATH = _ROOT / "config.yaml"


def _load_yaml() -> dict[str, Any]:
    with open(_CONFIG_PATH) as f:
        return yaml.safe_load(f)


_cfg = _load_yaml()


class _Scale:
    n_personas: int = _cfg["scale"]["n_personas"]
    n_charts: int = _cfg["scale"]["n_charts"]
    assignments_per_chart: int = _cfg["scale"]["assignments_per_chart"]


class _Models:
    persona_gen: str = _cfg["models"]["persona_gen"]
    preference_gen: str = _cfg["models"]["preference_gen"]
    rubric_gen: str = _cfg["models"]["rubric_gen"]
    tested_frontier: str = _cfg["models"]["tested_frontier"]
    tested_open: str = _cfg["models"]["tested_open"]
    judge: str = _cfg["models"]["judge"]
    embeddings: str = _cfg["models"]["embeddings"]


class _Inference:
    temperature_generation: float = _cfg["inference"]["temperature_generation"]
    temperature_preferences: float = _cfg["inference"]["temperature_preferences"]
    temperature_responses: float = _cfg["inference"]["temperature_responses"]
    temperature_judge: float = _cfg["inference"]["temperature_judge"]
    max_tokens: int = _cfg["inference"]["max_tokens"]


class _Paths:
    data_dir: Path = _ROOT / _cfg["paths"]["data_dir"]
    results_dir: Path = _ROOT / _cfg["paths"]["results_dir"]
    cache_dir: Path = _ROOT / _cfg["paths"]["cache_dir"]
    prompts_dir: Path = _ROOT / "prefvlm" / "prompts"
    charts_dir: Path = _ROOT / _cfg["paths"]["data_dir"] / "charts"
    preferences_dir: Path = _ROOT / _cfg["paths"]["data_dir"] / "preferences"
    rubrics_dir: Path = _ROOT / _cfg["paths"]["data_dir"] / "rubrics"
    responses_dir: Path = _ROOT / _cfg["paths"]["data_dir"] / "responses"
    judgments_dir: Path = _ROOT / _cfg["paths"]["data_dir"] / "judgments"
    logs_dir: Path = _ROOT / "logs"


class Config:
    scale = _Scale()
    models = _Models()
    inference = _Inference()
    paths = _Paths()
    seed: int = _cfg["seed"]
    openai_api_key: str = os.environ.get("OPENAI_API_KEY", "")

    @classmethod
    def validate(cls) -> None:
        if not cls.openai_api_key:
            raise EnvironmentError(
                "OPENAI_API_KEY is not set. Copy .env.example to .env and fill it in."
            )

    @classmethod
    def ensure_dirs(cls) -> None:
        """Create all output directories if they don't exist."""
        for attr in vars(cls.paths):
            path = getattr(cls.paths, attr)
            if isinstance(path, Path) and not path.suffix:
                path.mkdir(parents=True, exist_ok=True)
        (cls.paths.responses_dir / "frontier").mkdir(parents=True, exist_ok=True)
        (cls.paths.responses_dir / "qwen").mkdir(parents=True, exist_ok=True)
        cls.paths.logs_dir.mkdir(parents=True, exist_ok=True)


cfg = Config()
