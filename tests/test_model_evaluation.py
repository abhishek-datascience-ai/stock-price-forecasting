from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.model_evaluation import (
    PREDICTION_SPECS,
    calculate_metrics,
    evaluate_predictions,
    load_aligned_predictions,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def test_prediction_files_are_aligned() -> None:
    predictions = load_aligned_predictions()
    assert len(predictions) == len(PREDICTION_SPECS)
    assert len(predictions["Baseline"]) > 0


def test_fixed_origin_baseline_remains_constant() -> None:
    predictions = load_aligned_predictions()
    assert predictions["Baseline"]["Prediction"].nunique() == 1


def test_comparison_matches_prediction_files() -> None:
    predictions = load_aligned_predictions()
    calculated = evaluate_predictions(predictions).set_index("Model")
    stored = pd.read_csv(PROCESSED_DIR / "model_comparison.csv").set_index("Model")

    assert calculated.index.equals(stored.index)
    assert np.allclose(calculated.to_numpy(), stored.to_numpy(), rtol=0, atol=1e-9)


def test_evaluation_summary_matches_shared_window() -> None:
    predictions = load_aligned_predictions()
    reference = predictions["Baseline"]
    summary = json.loads(
        (PROCESSED_DIR / "model_evaluation_summary.json").read_text(encoding="utf-8")
    )

    assert summary["evaluation_rows"] == len(reference)
    assert summary["start_date"] == reference["Date"].min().strftime("%Y-%m-%d")
    assert summary["end_date"] == reference["Date"].max().strftime("%Y-%m-%d")
    assert summary["forecast_protocol"] == (
        "fixed-origin multi-step without test-period updates"
    )


def test_metrics_handle_zero_actual_values() -> None:
    metrics = calculate_metrics(pd.Series([0.0, 10.0]), pd.Series([2.0, 8.0]))
    assert metrics["MAE"] == 2.0
    assert metrics["MAPE (%)"] == 20.0
