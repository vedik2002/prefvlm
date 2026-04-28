"""Aggregate all judgments into summary tables and plots."""

import json
from collections import defaultdict
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from loguru import logger
from prefvlm.config import cfg


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def _load_judgments() -> pd.DataFrame:
    """Load all judgment files into a DataFrame."""
    rows = []
    for jf in sorted(cfg.paths.judgments_dir.glob("*.json")):
        j = json.loads(jf.read_text())
        # Derive model tag from filename suffix
        model_tag = "qwen" if jf.stem.endswith("_qwen") else "frontier"
        rows.append({
            "scenario_id":    j["scenario_id"],
            "persona_id":     j.get("persona_id", ""),
            "question_id":    j.get("question_id", ""),
            "condition":      j["condition"],
            "model_tag":      model_tag,
            "response_model": j.get("response_model", ""),
            "weighted_score": j["weighted_score"],
            "n_attributes":   j.get("n_attributes", 0),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def _condition_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Mean weighted score by condition × model_tag."""
    return (
        df.groupby(["model_tag", "condition"])["weighted_score"]
        .agg(["mean", "std", "min", "max", "count"])
        .round(3)
        .reset_index()
    )


def _oracle_lift(df: pd.DataFrame) -> pd.DataFrame:
    """Oracle lift = oracle mean - baseline mean, per model_tag."""
    rows = []
    for tag, g in df.groupby("model_tag"):
        means = g.groupby("condition")["weighted_score"].mean()
        rows.append({
            "model_tag":        tag,
            "baseline_mean":    round(means.get("baseline", float("nan")), 3),
            "oracle_mean":      round(means.get("oracle", float("nan")), 3),
            "wrong_persona_mean": round(means.get("wrong_persona", float("nan")), 3),
            "oracle_lift":      round(
                means.get("oracle", 0) - means.get("baseline", 0), 3
            ),
            "wrong_persona_lift": round(
                means.get("wrong_persona", 0) - means.get("baseline", 0), 3
            ),
        })
    return pd.DataFrame(rows)


def _per_scenario(df: pd.DataFrame) -> pd.DataFrame:
    return df[["scenario_id", "persona_id", "condition", "model_tag", "weighted_score"]].sort_values(
        ["scenario_id", "model_tag", "condition"]
    )


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

CONDITION_ORDER = ["baseline", "oracle", "wrong_persona"]
CONDITION_LABELS = {"baseline": "Baseline", "oracle": "Oracle", "wrong_persona": "Wrong persona"}
MODEL_COLORS = {"frontier": "#4C72B0", "qwen": "#DD8452"}


def _plot_condition_bars(df: pd.DataFrame, out_dir: Path) -> None:
    """Grouped bar chart: condition means for each model."""
    summary = _condition_summary(df)
    fig, ax = plt.subplots(figsize=(8, 5))

    tags = sorted(df["model_tag"].unique())
    x = range(len(CONDITION_ORDER))
    bar_w = 0.35

    for i, tag in enumerate(tags):
        sub = summary[summary["model_tag"] == tag].set_index("condition")
        means = [sub.loc[c, "mean"] if c in sub.index else 0 for c in CONDITION_ORDER]
        stds  = [sub.loc[c, "std"]  if c in sub.index else 0 for c in CONDITION_ORDER]
        offset = (i - len(tags) / 2 + 0.5) * bar_w
        bars = ax.bar(
            [xi + offset for xi in x], means, bar_w,
            label=tag.capitalize(), color=MODEL_COLORS.get(tag, "#888"),
            yerr=stds, capsize=4, alpha=0.88,
        )
        for bar, m in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.04,
                    f"{m:.2f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(list(x))
    ax.set_xticklabels([CONDITION_LABELS[c] for c in CONDITION_ORDER])
    ax.set_ylabel("Weighted preference score (1–5)")
    ax.set_title("Mean preference score by condition and model")
    ax.set_ylim(1, 5)
    ax.legend()
    ax.axhline(3, color="grey", linewidth=0.7, linestyle="--", alpha=0.5)
    plt.tight_layout()
    path = out_dir / "condition_bars.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.debug(f"Saved {path.name}")


def _plot_per_scenario(df: pd.DataFrame, out_dir: Path) -> None:
    """One panel per model: scatter/line of oracle vs baseline per scenario."""
    tags = sorted(df["model_tag"].unique())
    fig, axes = plt.subplots(1, len(tags), figsize=(6 * len(tags), 5), sharey=True)
    if len(tags) == 1:
        axes = [axes]

    for ax, tag in zip(axes, tags):
        sub = df[df["model_tag"] == tag]
        for cond, color, marker in [
            ("baseline",      "#4C72B0", "o"),
            ("oracle",        "#2ca02c", "^"),
            ("wrong_persona", "#d62728", "s"),
        ]:
            pts = sub[sub["condition"] == cond].sort_values("scenario_id")
            ax.plot(
                range(len(pts)), pts["weighted_score"].values,
                marker=marker, label=CONDITION_LABELS[cond],
                color=color, linewidth=1.5, markersize=7,
            )
        ax.set_title(f"{tag.capitalize()}")
        ax.set_xticks(range(len(sub["scenario_id"].unique())))
        ax.set_xticklabels(sorted(sub["scenario_id"].unique()), rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Weighted preference score")
        ax.set_ylim(1, 5)
        ax.axhline(3, color="grey", linewidth=0.7, linestyle="--", alpha=0.5)
        ax.legend()

    fig.suptitle("Per-scenario scores by condition", y=1.02)
    plt.tight_layout()
    path = out_dir / "per_scenario.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.debug(f"Saved {path.name}")


# ---------------------------------------------------------------------------
# Markdown summary
# ---------------------------------------------------------------------------

def _write_markdown(
    condition_df: pd.DataFrame,
    lift_df: pd.DataFrame,
    scenario_df: pd.DataFrame,
    out_path: Path,
) -> None:
    lines = [
        "# PrefVLM-MVP — Results Summary",
        "",
        "## Condition means by model",
        "",
        condition_df.to_markdown(index=False),
        "",
        "## Oracle lift over baseline",
        "",
        lift_df.to_markdown(index=False),
        "",
        "## Per-scenario scores",
        "",
        scenario_df.to_markdown(index=False),
        "",
        "## Plots",
        "",
        "- `plots/condition_bars.png` — grouped bar chart",
        "- `plots/per_scenario.png` — per-scenario lines",
    ]
    out_path.write_text("\n".join(lines))
    logger.debug(f"Saved {out_path.name}")


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------

def aggregate() -> dict:
    """Load all judgments, compute tables, save plots and summary.md."""
    df = _load_judgments()
    if df.empty:
        logger.warning("No judgment files found — nothing to aggregate")
        return {}

    logger.info(f"Loaded {len(df)} judgments across {df['scenario_id'].nunique()} scenarios")

    results_dir = cfg.paths.results_dir
    plots_dir = results_dir / "plots"
    results_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    condition_df  = _condition_summary(df)
    lift_df       = _oracle_lift(df)
    scenario_df   = _per_scenario(df)

    # Save CSV
    df.to_csv(results_dir / "scores.csv", index=False)
    condition_df.to_csv(results_dir / "condition_summary.csv", index=False)
    lift_df.to_csv(results_dir / "oracle_lift.csv", index=False)

    # Plots
    _plot_condition_bars(df, plots_dir)
    _plot_per_scenario(df, plots_dir)

    # Markdown
    _write_markdown(condition_df, lift_df, scenario_df, results_dir / "summary.md")

    logger.success(
        f"Aggregation complete — {len(df)} judgments, "
        f"{df['scenario_id'].nunique()} scenarios, "
        f"{df['model_tag'].nunique()} models"
    )
    return {
        "n_judgments": len(df),
        "n_scenarios": df["scenario_id"].nunique(),
        "condition_summary": condition_df.to_dict("records"),
        "oracle_lift": lift_df.to_dict("records"),
    }
