"""Single entry point for the PrefVLM-MVP pipeline.

Usage:
    python -m prefvlm.run_mvp --stage <stage>

Stages:
    personas       Generate 5 personas
    questions      Load 30 questions from ScienceQA
    scenarios      Assign personas to questions
    preferences    Instantiate preference profiles
    rubrics        Generate scoring rubrics
    frontier       Run GPT-4.1-mini inference (3 conditions)
    qwen-prepare   Bundle batch file for Colab
    qwen-ingest    Ingest Colab output
    judge          Score all responses
    aggregate      Compute summary tables and plots
    all            Run all stages in order (local stages only)
    validate       Test OpenAI connectivity
"""

from pathlib import Path

import click
from loguru import logger

from prefvlm.config import cfg
from prefvlm.logging_setup import setup_logging

STAGES = [
    "validate",
    "personas",
    "questions",
    "scenarios",
    "preferences",
    "rubrics",
    "frontier",
    "qwen-prepare",
    "qwen-ingest",
    "judge",
    "aggregate",
    "all",
]

LOCAL_STAGES = [
    "validate",
    "personas",
    "questions",
    "scenarios",
    "preferences",
    "rubrics",
    "frontier",
    "qwen-prepare",
    "qwen-ingest",
    "judge",
    "aggregate",
]


def run_validate() -> None:
    """Test OpenAI connectivity with a minimal API call."""
    from prefvlm.openai_client import chat_completion
    cfg.validate()
    logger.info("Testing OpenAI connectivity...")
    result = chat_completion(
        model=cfg.models.persona_gen,
        messages=[{"role": "user", "content": "Reply with exactly: OK"}],
        temperature=0.0,
        max_tokens=10,
    )
    logger.success(f"OpenAI API responded: {result!r}")
    click.echo(f"\n✓ OpenAI connectivity OK. Model response: {result!r}")


def run_personas() -> None:
    from prefvlm.personas.generate import generate_personas
    personas = generate_personas()
    click.echo(f"\n✓ Generated {len(personas)} personas. Saved to data/personas.json")
    for p in personas:
        click.echo(f"  • {p.name} ({p.age}, {p.occupation})")


def run_questions() -> None:
    from prefvlm.data.scienceqa import load_questions
    questions = load_questions()
    click.echo(f"\n✓ Loaded {len(questions)} questions. Saved to data/questions.json")
    for q in questions[:5]:
        click.echo(f"  • [{q['question_id']}] grade={q['grade']} {q['topic']:12s} | {q['question'][:55]}...")


def run_scenarios() -> None:
    from prefvlm.scenarios.assign import assign_scenarios
    scenarios = assign_scenarios()
    click.echo(f"\n✓ Assigned {len(scenarios)} scenarios. Saved to data/scenarios.json")


def run_preferences(limit: int = 0) -> None:
    from prefvlm.preferences.instantiate import instantiate_all
    instantiate_all(limit=limit)
    click.echo("\n✓ Preference instantiation complete.")


def run_rubrics(limit: int = 0) -> None:
    from prefvlm.preferences.rubrics import generate_all_rubrics
    generate_all_rubrics(limit=limit)
    click.echo("\n✓ Rubric generation complete.")


def run_frontier(limit: int = 0) -> None:
    from prefvlm.runners.frontier import run_frontier_inference
    run_frontier_inference(limit=limit)
    click.echo("\n✓ Frontier inference complete.")


def run_qwen_prepare(limit: int = 0) -> None:
    from prefvlm.runners.qwen_batch import prepare_batch
    prepare_batch(limit=limit)


def run_qwen_ingest() -> None:
    from prefvlm.runners.qwen_batch import ingest_batch
    ingest_batch()
    click.echo("\n✓ Qwen responses ingested.")


def run_judge(limit: int = 0) -> None:
    from prefvlm.judge.score import score_all
    score_all(limit=limit)
    click.echo("\n✓ Judging complete.")


def run_aggregate() -> None:
    from prefvlm.eval.aggregate import aggregate
    aggregate()
    click.echo("\n✓ Aggregation complete. See results/summary.md")


@click.command()
@click.option(
    "--stage",
    type=click.Choice(STAGES, case_sensitive=False),
    required=True,
    help="Pipeline stage to run.",
)
@click.option(
    "--limit",
    default=0,
    help="Process only N scenarios (0 = all). Useful for partial runs during dev.",
)
def cli(stage: str, limit: int) -> None:
    """PrefVLM-MVP pipeline runner."""
    setup_logging()
    cfg.ensure_dirs()

    dispatch = {
        "validate": lambda: run_validate(),
        "personas": lambda: run_personas(),
        "questions": lambda: run_questions(),
        "scenarios": lambda: run_scenarios(),
        "preferences": lambda: run_preferences(limit),
        "rubrics": lambda: run_rubrics(limit),
        "frontier": lambda: run_frontier(limit),
        "qwen-prepare": lambda: run_qwen_prepare(limit),
        "qwen-ingest": lambda: run_qwen_ingest(),
        "judge": lambda: run_judge(limit),
        "aggregate": lambda: run_aggregate(),
    }

    if stage == "all":
        for s in LOCAL_STAGES:
            if s in ("qwen-prepare", "qwen-ingest"):
                continue
            logger.info(f"=== Running stage: {s} ===")
            dispatch[s]()
    else:
        dispatch[stage]()


if __name__ == "__main__":
    cli()
