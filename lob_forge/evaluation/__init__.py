"""Backtesting, stylized fact validation, and performance metrics."""

from lob_forge.evaluation.lob_bench import (
    compute_conditional_stats,
    compute_wasserstein_metrics,
    run_lob_bench,
    train_discriminator,
)
from lob_forge.evaluation.regime_validation import (
    compare_regime_distributions,
    compute_regime_divergence,
    validate_regime_conditioning,
)
from lob_forge.evaluation.stylized_facts import (
    bid_ask_bounce_test,
    book_shape_test,
    market_impact_test,
    return_distribution_test,
    run_all_stylized_tests,
    spread_cdf_test,
    summary_figure,
    volatility_clustering_test,
)
from lob_forge.evaluation.validate_generator import validate_generator

__all__ = [
    # Stylized facts
    "run_all_stylized_tests",
    "summary_figure",
    "return_distribution_test",
    "volatility_clustering_test",
    "bid_ask_bounce_test",
    "spread_cdf_test",
    "book_shape_test",
    "market_impact_test",
    # LOB-Bench
    "run_lob_bench",
    "compute_wasserstein_metrics",
    "train_discriminator",
    "compute_conditional_stats",
    # Regime validation
    "validate_regime_conditioning",
    "compare_regime_distributions",
    "compute_regime_divergence",
    # Pipeline
    "validate_generator",
]
