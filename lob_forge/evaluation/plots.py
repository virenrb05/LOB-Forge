"""Publication-ready matplotlib figures for LOB execution evaluation.

Provides :func:`generate_all_plots` which produces 6 figures and saves them as
PNG files to a configurable output directory.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")  # non-interactive backend; safe for scripts / CI

if TYPE_CHECKING:
    pass

_STYLE = "seaborn-v0_8-whitegrid"
_FIGSIZE = (8, 5)
_DPI = 150
_AGENT_ORDER = ["dqn", "twap", "vwap", "almgren_chriss", "random"]
_AGENT_LABELS = {
    "dqn": "DQN",
    "twap": "TWAP",
    "vwap": "VWAP",
    "almgren_chriss": "Almgren-Chriss",
    "random": "Random",
}
_COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]


# ---------------------------------------------------------------------------
# Individual plot helpers
# ---------------------------------------------------------------------------


def _agent_cost_comparison(comparison_dict: dict, output_dir: Path) -> Path:
    """Bar chart: mean IS (episode_cost) per agent with error bars (std)."""
    agents = [a for a in _AGENT_ORDER if a in comparison_dict]
    means = []
    stds = []
    for agent in agents:
        costs = [r.episode_cost for r in comparison_dict[agent]["results"]]
        means.append(float(np.mean(costs)))
        stds.append(float(np.std(costs, ddof=0)))

    labels = [_AGENT_LABELS[a] for a in agents]
    colors = _COLORS[: len(agents)]

    with plt.style.context(_STYLE):
        fig, ax = plt.subplots(figsize=_FIGSIZE, dpi=_DPI)
        x = np.arange(len(labels))
        bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors, edgecolor="white")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_xlabel("Agent")
        ax.set_ylabel("Mean Episode Cost")
        ax.set_title("Agent Cost Comparison (Mean IS ± Std)")
        ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=8)
        plt.tight_layout()
        out = output_dir / "agent_cost_comparison.png"
        fig.savefig(out, dpi=_DPI)
        plt.close(fig)
    return out


def _is_sharpe_comparison(comparison_dict: dict, output_dir: Path) -> Path:
    """Bar chart: IS Sharpe per agent."""
    from lob_forge.evaluation.metrics import compute_is_sharpe  # noqa: PLC0415

    agents = [a for a in _AGENT_ORDER if a in comparison_dict]
    sharpes = [compute_is_sharpe(comparison_dict[a]["results"]) for a in agents]
    labels = [_AGENT_LABELS[a] for a in agents]
    colors = _COLORS[: len(agents)]

    with plt.style.context(_STYLE):
        fig, ax = plt.subplots(figsize=_FIGSIZE, dpi=_DPI)
        x = np.arange(len(labels))
        bars = ax.bar(x, sharpes, color=colors, edgecolor="white")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_xlabel("Agent")
        ax.set_ylabel("IS Sharpe")
        ax.set_title("IS Sharpe Ratio by Agent")
        ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=8)
        plt.tight_layout()
        out = output_dir / "is_sharpe_comparison.png"
        fig.savefig(out, dpi=_DPI)
        plt.close(fig)
    return out


def _slippage_vs_twap(comparison_dict: dict, output_dir: Path) -> Path:
    """Bar chart: (agent_cost - twap_cost) / twap_cost for non-TWAP agents."""
    from lob_forge.evaluation.metrics import compute_slippage_vs_twap  # noqa: PLC0415

    twap_results = comparison_dict.get("twap", {}).get("results", [])
    non_twap = [a for a in _AGENT_ORDER if a != "twap" and a in comparison_dict]
    slippages = [
        compute_slippage_vs_twap(comparison_dict[a]["results"], twap_results)
        for a in non_twap
    ]
    labels = [_AGENT_LABELS[a] for a in non_twap]
    color_map = {a: c for a, c in zip(_AGENT_ORDER, _COLORS, strict=False)}
    colors = [color_map[a] for a in non_twap]

    with plt.style.context(_STYLE):
        fig, ax = plt.subplots(figsize=_FIGSIZE, dpi=_DPI)
        x = np.arange(len(labels))
        bars = ax.bar(x, slippages, color=colors, edgecolor="white")
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_xlabel("Agent")
        ax.set_ylabel("Slippage vs TWAP (negative = better)")
        ax.set_title("Slippage vs TWAP Baseline")
        ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=8)
        plt.tight_layout()
        out = output_dir / "slippage_vs_twap.png"
        fig.savefig(out, dpi=_DPI)
        plt.close(fig)
    return out


def _cumulative_cost_curve(comparison_dict: dict, output_dir: Path) -> Path:
    """Line plot: cumulative episode cost across episodes for each agent."""
    agents = [a for a in _AGENT_ORDER if a in comparison_dict]
    color_map = {a: c for a, c in zip(_AGENT_ORDER, _COLORS, strict=False)}

    with plt.style.context(_STYLE):
        fig, ax = plt.subplots(figsize=_FIGSIZE, dpi=_DPI)
        for agent in agents:
            costs = [r.episode_cost for r in comparison_dict[agent]["results"]]
            cum = np.cumsum(costs)
            episodes = np.arange(1, len(cum) + 1)
            ax.plot(
                episodes,
                cum,
                label=_AGENT_LABELS[agent],
                color=color_map[agent],
                marker="o",
                markersize=4,
            )
        ax.set_xlabel("Episode")
        ax.set_ylabel("Cumulative Cost")
        ax.set_title("Cumulative Execution Cost Across Episodes")
        ax.legend(loc="upper left")
        plt.tight_layout()
        out = output_dir / "cumulative_cost_curve.png"
        fig.savefig(out, dpi=_DPI)
        plt.close(fig)
    return out


def _action_distribution(comparison_dict: dict, output_dir: Path) -> Path:
    """Stacked bar chart: action frequencies for DQN, TWAP, and Random."""
    from lob_forge.executor import ACTION_NAMES  # noqa: PLC0415

    focus_agents = ["dqn", "twap", "random"]
    present = [a for a in focus_agents if a in comparison_dict]

    n_actions = 7
    x = np.arange(n_actions)
    width = 0.25
    color_map = {a: c for a, c in zip(_AGENT_ORDER, _COLORS, strict=False)}

    # Count action frequencies
    freq: dict[str, np.ndarray] = {}
    for agent in present:
        all_actions: list[int] = []
        for r in comparison_dict[agent]["results"]:
            all_actions.extend(r.actions)
        counts = np.zeros(n_actions, dtype=float)
        for a in all_actions:
            if 0 <= a < n_actions:
                counts[a] += 1
        total = counts.sum()
        freq[agent] = counts / total if total > 0 else counts

    with plt.style.context(_STYLE):
        fig, ax = plt.subplots(figsize=_FIGSIZE, dpi=_DPI)
        n_agents = len(present)
        offsets = np.linspace(
            -(n_agents - 1) * width / 2, (n_agents - 1) * width / 2, n_agents
        )
        for i, agent in enumerate(present):
            ax.bar(
                x + offsets[i],
                freq[agent],
                width=width,
                label=_AGENT_LABELS[agent],
                color=color_map[agent],
                edgecolor="white",
            )
        ax.set_xticks(x)
        action_labels = (
            list(ACTION_NAMES)
            if hasattr(ACTION_NAMES, "__iter__")
            else [str(i) for i in range(n_actions)]
        )
        ax.set_xticklabels(action_labels[:n_actions], rotation=30, ha="right")
        ax.set_xlabel("Action")
        ax.set_ylabel("Relative Frequency")
        ax.set_title("Action Distribution: DQN vs TWAP vs Random")
        ax.legend()
        plt.tight_layout()
        out = output_dir / "action_distribution.png"
        fig.savefig(out, dpi=_DPI)
        plt.close(fig)
    return out


def _training_loss_curve(output_dir: Path) -> Path:
    """Line plot of training loss; placeholder when log CSV is absent."""
    log_path = Path("checkpoints") / "training_log.csv"

    with plt.style.context(_STYLE):
        fig, ax = plt.subplots(figsize=_FIGSIZE, dpi=_DPI)

        if log_path.exists():
            import csv  # noqa: PLC0415

            steps: list[float] = []
            losses: list[float] = []
            with open(log_path, newline="") as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    # Support common column names
                    step_val = row.get("step") or row.get("steps") or row.get("epoch")
                    loss_val = (
                        row.get("loss") or row.get("td_loss") or row.get("train_loss")
                    )
                    if step_val is not None and loss_val is not None:
                        try:
                            steps.append(float(step_val))
                            losses.append(float(loss_val))
                        except ValueError:
                            continue
            if steps:
                ax.plot(steps, losses, color=_COLORS[0], linewidth=1.5)
                ax.set_xlabel("Step")
                ax.set_ylabel("TD Loss")
                ax.set_title("Training Loss Curve")
            else:
                ax.text(
                    0.5,
                    0.5,
                    "No parseable data in training log",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=14,
                    color="gray",
                )
                ax.set_title("Training Loss Curve")
        else:
            ax.text(
                0.5,
                0.5,
                "No training log found",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=14,
                color="gray",
            )
            ax.set_title("Training Loss Curve (placeholder)")

        plt.tight_layout()
        out = output_dir / "training_loss_curve.png"
        fig.savefig(out, dpi=_DPI)
        plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_all_plots(
    comparison_dict: dict,
    output_dir: str | Path = "outputs/plots",
) -> list[Path]:
    """Generate all 6 publication-ready plots and save them as PNG files.

    Parameters
    ----------
    comparison_dict : dict
        Agent comparison data.  Structure::

            {
                "dqn":            {"results": list[ExecutionResult], ...},
                "twap":           {"results": list[ExecutionResult], ...},
                "vwap":           {"results": list[ExecutionResult], ...},
                "almgren_chriss": {"results": list[ExecutionResult], ...},
                "random":         {"results": list[ExecutionResult], ...},
                ...  # optional extra keys (e.g. "dqn_beats_twap") are ignored
            }

        This matches the output of
        :func:`~lob_forge.executor.evaluate.compare_to_baselines`.

    output_dir : str | Path
        Directory where PNG files are saved.  Created if it does not exist.
        Defaults to ``"outputs/plots"``.

    Returns
    -------
    list[Path]
        Absolute paths to the 6 saved PNG files in generation order.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths: list[Path] = []
    paths.append(_agent_cost_comparison(comparison_dict, output_dir))
    paths.append(_is_sharpe_comparison(comparison_dict, output_dir))
    paths.append(_slippage_vs_twap(comparison_dict, output_dir))
    paths.append(_cumulative_cost_curve(comparison_dict, output_dir))
    paths.append(_action_distribution(comparison_dict, output_dir))
    paths.append(_training_loss_curve(output_dir))

    return paths


__all__ = ["generate_all_plots"]
