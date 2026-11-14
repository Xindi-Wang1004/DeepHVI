import json

from deephvi.cli import main
from deephvi.data.legacy import write_pairs_csv
from deephvi.data.synthetic import generate_synthetic_records


def test_tiny_three_fold_pipeline_and_evaluation(tmp_path):
    pairs = write_pairs_csv(generate_synthetic_records(n_pairs=24), tmp_path / "pairs.csv")
    output = tmp_path / "run"
    assert main(["reproduce", "--config", "configs/demo_cpu.yaml", "--pairs", str(pairs),
                 "--out", str(output), "--limit", "18", "--epochs", "1", "--device", "cpu"]) == 0
    assert len(json.loads((output / "checkpoint_map.json").read_text())) == 3
    assert main(["evaluate", "--checkpoint-map", str(output / "checkpoint_map.json"), "--fold", "all",
                 "--bundle", str(output / "features"), "--splits", str(output / "splits"),
                 "--out", str(tmp_path / "evaluation"), "--device", "cpu"]) == 0
    assert main(["evaluate", "--checkpoint", str(output / "fold0/checkpoints/best.pt"), "--fold", "1",
                 "--bundle", str(output / "features"), "--splits", str(output / "splits"), "--device", "cpu"]) == 2
