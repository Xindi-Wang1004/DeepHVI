"""CLI workflow acceptance with isolated HTTP fixtures and a demo-only encoder."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
from urllib.error import URLError

import numpy as np
import pytest

from deephvi.cli import main
from deephvi.data import reference_download as download
from deephvi.data.sequence_backend import HashSequenceBackend


@pytest.fixture
def workflow(tmp_path, monkeypatch):
    calls, encoded = [], []
    human = gzip.compress(b">NP_1.1\nACDEFGHIK\n>XP_2.1\nACDEFGHIK\n>XP_3.1\nYYYYYYYYY\n")
    md5 = hashlib.md5(human).hexdigest()
    replies = {
        download.HUMAN_BASE + "md5checksums.txt": (
            f"{md5}  ./{download.HUMAN_FILE}\n".encode(), {}),
        download.HUMAN_BASE + download.HUMAN_FILE: (human, {}),
        download.VIRAL_URL: (
            b">sp|P12345|FIRST\nACDEFGHIK\n>tr|A0A123|SAME\nACDEFGHIK\n>tr|A0A124|OTHER\nYYYYYYYYY\n",
            {"x-uniprot-release": "ISOLATED_TEST_RELEASE", "x-total-results": "3"},
        ),
    }

    def fetch(url, path):
        calls.append(url)
        body, headers = replies[url]
        path.write_bytes(body)
        return headers

    class Encoder(HashSequenceBackend):
        def encode(self, sequences, *, batch_size=None):
            encoded.append(list(sequences))
            return super().encode(sequences, batch_size=batch_size)

    backend = Encoder()
    monkeypatch.setattr(download, "_fetch", fetch)
    monkeypatch.setattr("deephvi.data.sequence_backend.build_backend", lambda *a, **kw: backend)
    pairs = tmp_path / "pairs.csv"
    pairs.write_text("human_seq,viral_seq\nACDEFGHIK,YYYYYYYYY\n")
    common = ["--config", "configs/demo_cpu.yaml", "--pairs", str(pairs), "--device", "cpu",
              "--reference-dir", str(tmp_path / "references"), "--log-level", "WARNING"]
    return tmp_path, common, calls, encoded, backend


@pytest.mark.parametrize("source", ["human", "viral"])
@pytest.mark.parametrize("command", ["retrieve", "generate"])
def test_one_command_downloads_prepares_indexes_and_reuses(workflow, source, command):
    root, common, calls, encoded, backend = workflow
    output = root / "results.json"
    args = [command, *common, "--reference-source", source, "--top-k", "1", "--out", str(output)]
    assert main(args) == 0
    result = json.loads(output.read_text())
    assert result["target"] == source
    assert result["reference"]["source"] == source
    assert result["reference"]["statistics"]["unique_sequences"] == 2
    assert result["reference"]["statistics"]["accessions"] == 3
    assert len(result["results"]) == 1
    expected = "ACDEFGHIK" if source == "human" else "YYYYYYYYY"
    assert result["results"][0]["hit_sequence"] == expected
    index = root / "references/indices" / source
    assert len(list(index.iterdir())) == 1
    index_path = next(index.iterdir())
    assert result["index"] == str(index_path)
    completion = index_path / "pipeline.json"
    receipt_before = completion.read_bytes(), completion.stat().st_mtime_ns
    assert len(encoded) == 2  # One gallery encoding and one query encoding.
    assert len(encoded[0]) == 2
    network_calls = len(calls)
    # The same command validates and reuses every persisted stage offline.
    assert main(args) == 0
    assert len(calls) == network_calls
    assert len(encoded) == 3 and encoded[-1] == [expected]
    assert (completion.read_bytes(), completion.stat().st_mtime_ns) == receipt_before
    # Explicit --index also works, with managed-index integrity checks.
    assert main([command, *common, "--index", str(index_path), "--target", source,
                 "--out", str(root / "explicit.json")]) == 0
    assert len(calls) == network_calls
    assert backend.name == "hash"  # No biological model claims from this fixture.


def test_automatic_reference_validates_arguments_before_downloading(workflow):
    root, common, calls, encoded, _ = workflow
    assert main(["retrieve", *common, "--reference-source", "human", "--target", "viral"]) == 2
    assert main(["retrieve", *common, "--reference-source", "viral", "--top-k", "0"]) == 2
    assert main(["retrieve", *common]) == 2
    for alternative in ("--reference", "--index"):
        with pytest.raises(SystemExit) as error:
            main(["retrieve", *common, "--reference-source", "viral", alternative, str(root / "file")])
        assert error.value.code == 2
    assert not calls and not encoded


def test_pipeline_rejects_source_and_index_corruption_and_can_rebuild(workflow):
    root, common, calls, encoded, _ = workflow
    output = root / "results.json"
    args = ["retrieve", *common, "--reference-source", "viral", "--out", str(output)]
    assert main(args) == 0
    index = next((root / "references/indices/viral").iterdir())
    embeddings = index / "embeddings.npy"
    # Valid shape and finite numbers still must fail the persisted fingerprint.
    array = np.load(embeddings)
    array[0, 0] += 1
    np.save(embeddings, array)
    assert main(args) == 2
    assert main(["retrieve", *common, "--index", str(index), "--out", str(output)]) == 2
    assert main([*args, "--overwrite"]) == 0
    assert len(calls) == 1
    assert len(encoded) == 4
    raw = root / "references/raw/viral" / download.VIRAL_FILE
    raw.write_bytes(b"corruption")
    assert main(args) == 2
    assert len(calls) == 1  # Completed corrupted data is rejected, not replaced silently.


def test_encoder_identity_selects_separate_automatic_index(workflow):
    root, common, calls, encoded, backend = workflow
    args = ["generate", *common, "--reference-source", "viral", "--out", str(root / "results.json")]
    assert main(args) == 0
    backend.k = 2
    assert main(args) == 0
    assert len(list((root / "references/indices/viral").iterdir())) == 2
    assert len(calls) == 1 and len(encoded) == 4


def test_explicit_index_output_rejects_different_encoder_without_overwrite(workflow):
    root, common, calls, _, backend = workflow
    args = ["retrieve", *common, "--reference-source", "viral", "--index-out", str(root / "index"),
            "--out", str(root / "results.json")]
    assert main(args) == 0
    backend.k = 2
    assert main(args) == 2
    assert main([*args, "--overwrite"]) == 0
    assert len(calls) == 1


def test_interrupted_index_build_restarts_without_downloading_again(workflow, monkeypatch):
    root, common, calls, encoded, backend = workflow
    original = backend.encode

    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(backend, "encode", interrupt)
    args = ["generate", *common, "--reference-source", "viral", "--out", str(root / "results.json")]
    assert main(args) == 130
    index = next((root / "references/indices/viral").iterdir())
    assert (index / "pipeline-progress.json").exists()
    assert not (index / "pipeline.json").exists()
    monkeypatch.setattr(backend, "encode", original)
    assert main(args) == 0
    assert len(calls) == 1 and len(encoded) == 2
    assert not (index / "pipeline-progress.json").exists()


def test_managed_index_missing_completion_cannot_be_used_directly(workflow):
    root, common, _, _, _ = workflow
    args = ["generate", *common, "--reference-source", "viral", "--out", str(root / "results.json")]
    assert main(args) == 0
    index = next((root / "references/indices/viral").iterdir())
    (index / "pipeline.json").unlink()
    assert main(["generate", *common, "--index", str(index), "--out", str(root / "explicit.json")]) == 2
    assert main(args) == 2
    assert main([*args, "--overwrite"]) == 0


def test_interrupted_index_publication_is_recoverable(workflow, monkeypatch):
    from deephvi.inference import reference_pipeline

    root, common, calls, _, _ = workflow
    original = reference_pipeline.atomic_json

    def interrupt(path, value):
        if path.name == "pipeline.json":
            raise KeyboardInterrupt
        original(path, value)

    monkeypatch.setattr(reference_pipeline, "atomic_json", interrupt)
    args = ["retrieve", *common, "--reference-source", "viral", "--out", str(root / "results.json")]
    assert main(args) == 130
    index = next((root / "references/indices/viral").iterdir())
    assert (index / "metadata.json").exists()
    assert not (index / "pipeline.json").exists()
    monkeypatch.setattr(reference_pipeline, "atomic_json", original)
    assert main(args) == 0
    assert len(calls) == 1


def test_custom_fasta_and_index_workflows_remain_available(workflow):
    import pandas as pd

    root, common, calls, _, _ = workflow
    fasta = root / "custom.fasta"
    fasta.write_text(">custom_first\nACDEFGHIK\n>custom_second\nYYYYYYYYY\n")
    index, output = root / "custom-index", root / "results.csv"
    assert main(["generate", *common, "--reference", str(fasta), "--index-out", str(index),
                 "--top-k", "1", "--out", str(output)]) == 0
    assert pd.read_csv(output)["hit_id"].tolist() == ["custom_second"]
    assert main(["retrieve", *common, "--index", str(index), "--target", "human",
                 "--top-k", "1", "--out", str(output)]) == 0
    assert pd.read_csv(output)["hit_id"].tolist() == ["custom_first"]
    assert not calls


def test_public_reference_index_side_cannot_be_swapped(workflow):
    root, common, _, _, _ = workflow
    args = ["retrieve", *common, "--reference-source", "human", "--out", str(root / "results.json")]
    assert main(args) == 0
    index = next((root / "references/indices/human").iterdir())
    assert main(["retrieve", *common, "--index", str(index), "--target", "viral"]) == 2


def test_download_errors_return_cli_failure(workflow, monkeypatch):
    _, common, _, _, _ = workflow

    def fail(*args, **kwargs):
        raise URLError("isolated upstream failure")

    monkeypatch.setattr(download, "_fetch", fail)
    assert main(["retrieve", *common, "--reference-source", "viral"]) == 2


def test_reproduction_pins_both_galleries_without_changing_training(workflow, monkeypatch):
    import random
    import shutil

    import torch

    from deephvi.checkpoints import load_checkpoint
    from deephvi.data.features import FeatureBundle
    from deephvi.data.legacy import write_pairs_csv
    from deephvi.data.splits import load_splits
    from deephvi.data.synthetic import generate_synthetic_records

    root, common, calls, _, backend = workflow
    original_encode = backend.encode

    def consuming_encoder(*args, **kwargs):
        # Real encoders may consume RNGs during initialization or preprocessing.
        torch.rand(1)
        np.random.random()
        random.random()
        return original_encode(*args, **kwargs)

    monkeypatch.setattr(backend, "encode", consuming_encoder)
    pairs = write_pairs_csv(generate_synthetic_records(n_pairs=18), root / "training.csv")
    enabled, baseline = root / "with-references", root / "baseline"
    args = ["reproduce", *common, "--pairs", str(pairs), "--epochs", "1"]
    assert main([*args, "--reference-source", "both", "--out", str(enabled)]) == 0
    references = json.loads((enabled / "references.json").read_text())
    assert set(references) == {"human", "viral"}
    assert len(calls) == 3
    assert main([*args, "--out", str(baseline)]) == 0
    assert not (baseline / "references.json").exists()
    assert FeatureBundle.load(enabled / "features").fingerprint == FeatureBundle.load(baseline / "features").fingerprint
    assert load_splits(enabled / "splits").fingerprint == load_splits(baseline / "splits").fingerprint
    for fold in range(3):
        for name in ("best.pt", "best_pretrain.pt"):
            actual = load_checkpoint(enabled / f"fold{fold}/checkpoints/{name}")
            expected = load_checkpoint(baseline / f"fold{fold}/checkpoints/{name}")
            assert actual["reference_libraries"] == references
            assert actual["config"]["run"]["reference_libraries"] == references
            assert all(torch.equal(value, expected["state_dict"][key])
                       for key, value in actual["state_dict"].items())
    # Use only the trained checkpoint and target; no source/index arguments needed.
    checkpoint = enabled / "fold0/checkpoints/best.pt"
    for side in ("human", "viral"):
        output = root / (side + "-from-training.json")
        assert main(["generate", *common, "--checkpoint", str(checkpoint),
                     "--target", side, "--out", str(output)]) == 0
        result = json.loads(output.read_text())
        assert result["index"] == references[side]["index"]
        assert result["reference"]["source"] == side
    assert len(calls) == 3
    relocated = root / "relocated-viral"
    shutil.copytree(references["viral"]["index"], relocated)
    assert main(["retrieve", *common, "--checkpoint", str(checkpoint), "--index", str(relocated),
                 "--target", "viral", "--out", str(root / "relocated.json")]) == 0
    receipt = json.loads((relocated / "pipeline.json").read_text())
    receipt["unexpected_change"] = True
    (relocated / "pipeline.json").write_text(json.dumps(receipt))
    assert main(["retrieve", *common, "--checkpoint", str(checkpoint), "--index", str(relocated),
                 "--target", "viral", "--out", str(root / "mismatched.json")]) == 2
    # Prediction stays independent unless gallery preparation is requested.
    raw = root / "references/raw/viral" / download.VIRAL_FILE
    original = raw.read_bytes()
    raw.write_bytes(b"damaged library")
    assert main(["predict", *common, "--checkpoint", str(checkpoint),
                 "--out", str(root / "predictions.csv")]) == 0
    raw.write_bytes(original)
    assert main(["predict", *common, "--checkpoint", str(checkpoint),
                 "--reference-source", "both", "--out", str(root / "predictions-with-library.csv")]) == 0
    sidecar = json.loads((root / "predictions-with-library.csv.references.json").read_text())
    assert sidecar == references
    # No change to predicted probabilities or input rows from optional preparation.
    assert (root / "predictions.csv").read_bytes() == (root / "predictions-with-library.csv").read_bytes()
    assert len(calls) == 3


def test_training_reference_config_and_cli_precedence(workflow):
    from deephvi.config import load_config
    from deephvi.training.references import prepare_training_references

    root, common, calls, _, backend = workflow
    config = load_config("configs/demo_cpu.yaml", [
        "references.sources=[human, viral]", f"references.dir={root / 'yaml-references'}",
    ])
    records = prepare_training_references(config, backend=backend)
    assert set(records) == {"human", "viral"}
    assert all("yaml-references" in record["index"] for record in records.values())
    assert len(calls) == 3
    # CLI values override an explicit YAML profile for classification preparation.
    from deephvi.cli import _config, _reference_preparation_config, build_parser

    args = build_parser().parse_args([
        "predict", *common, "--set", "references.sources=[human]",
        "--set", f"references.dir={root / 'unused'}", "--reference-source", "both",
    ])
    actual = _reference_preparation_config(_config(args), args)
    assert actual["references"]["sources"] == ["human", "viral"]
    assert actual["references"]["dir"] == str(root / "references")
    # Retrieval can select a configured source/dir without repeating CLI options.
    pair_path = common[common.index("--pairs") + 1]
    assert main(["retrieve", "--config", "configs/demo_cpu.yaml", "--pairs", pair_path,
                 "--set", "references.sources=[human, viral]",
                 "--set", f"references.dir={root / 'yaml-references'}", "--target", "human",
                 "--out", str(root / "configured.json")]) == 0
    result = json.loads((root / "configured.json").read_text())
    assert result["index"] == records["human"]["index"]
    assert len(calls) == 3


@pytest.mark.parametrize("value", ["human", "[human, human]", "[other]", "[{}]"])
def test_invalid_training_reference_config_is_rejected(value):
    from deephvi.config import load_config

    with pytest.raises(ValueError, match="references.sources"):
        load_config(overrides=["references.sources=" + value])


def test_staged_training_inherits_pinned_references_and_resume_checks_them(workflow, capsys):
    from deephvi.checkpoints import load_checkpoint
    from deephvi.data.legacy import write_pairs_csv
    from deephvi.data.synthetic import generate_synthetic_records

    root, common, calls, _, _ = workflow
    pairs = write_pairs_csv(generate_synthetic_records(n_pairs=18), root / "training.csv")
    pair_options = ["--config", "configs/demo_cpu.yaml", "--pairs", str(pairs), "--device", "cpu"]
    assert main(["prepare-features", *pair_options, "--out", str(root / "features")]) == 0
    assert main(["make-splits", *pair_options, "--out", str(root / "splits")]) == 0
    training = ["--config", "configs/demo_cpu.yaml", "--bundle", str(root / "features"),
                "--splits", str(root / "splits"), "--epochs", "1", "--device", "cpu",
                "--set", f"run.output_dir={root / 'training-runs'}", "--log-level", "WARNING"]
    capsys.readouterr()
    assert main(["train-pretrain", *training, "--reference-source", "both",
                 "--reference-dir", str(root / "references")]) == 0
    pretrain = json.loads(capsys.readouterr().out)["best_checkpoint"]
    refs = load_checkpoint(pretrain)["reference_libraries"]
    assert set(refs) == {"human", "viral"}
    assert main(["train-classifier", *training, "--pretrain-checkpoint", pretrain]) == 0
    classifier = json.loads(capsys.readouterr().out)["best_checkpoint"]
    payload = load_checkpoint(classifier)
    assert payload["reference_libraries"] == refs
    run = root / Path(classifier).parent.parent
    assert json.loads((run / "references.json").read_text()) == refs
    # A new run inherits the recorded index and keeps it pinned during resume.
    last = str(Path(classifier).with_name("last.pt"))
    assert main(["train-classifier", *training, "--resume", last]) == 0
    resumed = json.loads(capsys.readouterr().out)["best_checkpoint"]
    assert load_checkpoint(resumed)["reference_libraries"] == refs
    assert main(["train-classifier", *training, "--resume", last, "--reference-source", "human",
                 "--reference-dir", str(root / "references")]) == 2
    assert len(calls) == 3
