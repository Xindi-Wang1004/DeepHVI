from __future__ import annotations

import json
from pathlib import Path

from deephvi.cli import main


def test_cpu_demo_runs_end_to_end(tmp_path: Path):
    output = tmp_path / "demo"
    code = main(
        [
            "demo",
            "--backend",
            "hash",
            "--n-pairs",
            "30",
            "--epochs",
            "1",
            "--device",
            "cpu",
            "--out",
            str(output),
            "--log-level",
            "WARNING",
        ]
    )
    assert code == 0
    summary = json.loads((output / "demo_summary.json").read_text(encoding="utf-8"))
    assert summary["backend"] == "hash"
    assert set(summary["test_metrics"]) >= {"accuracy", "precision", "recall", "f1", "roc_auc"}
    for relative in (
        "checkpoints/best.pt",
        "checkpoints/best_pretrain.pt",
        "features/manifest.json",
        "splits/splits.json",
        "predictions/test_predictions.csv",
        "predictions/new_pairs.csv",
        "metrics/test_metrics.json",
        "retrieval/reference_index/embeddings.npy",
        "retrieval/candidates.json",
    ):
        assert (output / relative).exists(), relative


def test_config_show_reflects_overrides(capsys):
    code = main(
        [
            "--config",
            "configs/demo_cpu.yaml",
            "--set",
            "train_classifier.epochs=7",
            "config",
            "show",
        ]
    )
    assert code == 0
    printed = capsys.readouterr().out
    assert "epochs: 7" in printed
    assert "backend: hash" in printed
