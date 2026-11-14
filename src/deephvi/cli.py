"""Command-line interface for DeepHVI."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml

from . import __version__
from .checkpoints import CheckpointError, load_checkpoint
from .config import (
    apply_overrides,
    config_for_json,
    deep_merge,
    load_config,
)
from .data.aaindex import AAindexError
from .data.dataset import build_dataloader
from .data.download import ArchiveError, prepare_dataset
from .data.esm import ESMBackendError
from .data.features import (
    FeatureBundle,
    FeatureBundleError,
    prepare_feature_bundle,
    verify_bundle_encoding,
)
from .data.legacy import (
    PairTableError,
    inspect_table,
    read_pair_table,
    read_training_pairs,
    write_pairs_csv,
)
from .data.negatives import ensure_negatives
from .data.reference_download import ReferenceDataError
from .data.schema import PairRecord
from .data.server import SERVER_MAX_LENGTH
from .data.splits import LeakageError, SplitError, load_splits, make_splits, save_splits
from .data.synthetic import generate_synthetic_records
from .doctor import collect_doctor_report, write_missing_resources
from .evaluation.evaluate import evaluate_loader
from .evaluation.metrics import aggregate_metrics, format_metrics
from .evaluation.report import write_markdown_report, write_metrics, write_predictions
from .inference.legacy_export import LegacyExportError, export_legacy_checkpoint
from .inference.predict import (
    check_backend_matches,
    load_model_for_prediction,
    predict_bundle,
    predict_records,
)
from .inference.retrieve import (
    ReferenceIndex,
    RetrievalError,
    build_index_from_sequences,
    build_reference_index,
    read_fasta,
    retrieve_for_records,
)
from .io import DataFormatError, write_json
from .logging_utils import get_logger, setup_logging
from .models.registry import build_model, load_model_from_checkpoint, verify_feature_spec
from .paths import RunDirectory
from .seeding import resolve_device, seed_everything
from .training.trainer import train_classifier, train_pretrain

MODALITY_KEYS = ("human_esm", "viral_esm", "human_aa", "viral_aa")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--resources-dir", type=Path, default=argparse.SUPPRESS, help="Unpacked versioned resource bundle")
    parser.add_argument("--config", type=Path, default=argparse.SUPPRESS, help="YAML profile")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=argparse.SUPPRESS,
        metavar="KEY=VALUE",
        help="Override a configuration value; repeatable",
    )
    parser.add_argument("--seed", type=int, default=argparse.SUPPRESS, help="Override all seeds")
    parser.add_argument("--device", default=argparse.SUPPRESS, help="auto, cpu or cuda")
    parser.add_argument("--log-level", default=argparse.SUPPRESS, help="DEBUG, INFO, WARNING")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="deephvi",
        description="DeepHVI: multimodal human-virus protein interaction prediction",
    )
    parser.add_argument("--version", action="version", version=f"deephvi {__version__}")
    _add_common(parser)
    subparsers = parser.add_subparsers(dest="command", required=True)

    demo = subparsers.add_parser("demo", help="Run the full CPU pipeline on synthetic data")
    _add_common(demo)
    demo.add_argument("--n-pairs", type=int, default=64)
    demo.add_argument("--positive-fraction", type=float, default=2 / 3)
    demo.add_argument("--backend", default="hash", choices=["hash", "esm2"])
    demo.add_argument("--epochs", type=int, default=None, help="Override both training stages")
    demo.add_argument("--out", type=Path, default=Path("runs/demo"))
    demo.set_defaults(func=cmd_demo)

    prep = subparsers.add_parser("prepare-data", help="Extract data/archives/data.7z")
    _add_common(prep)
    prep.add_argument("--archive", type=Path, default=Path("data/archives/data.7z"))
    prep.add_argument("--dest", type=Path, default=Path("runs/extracted"))
    prep.add_argument("--overwrite", action="store_true")
    prep.set_defaults(func=cmd_prepare_data)

    inspect = subparsers.add_parser("inspect-data", help="Inspect a pair CSV without loading it fully")
    _add_common(inspect)
    inspect.add_argument("--input", type=Path, required=True)
    inspect.add_argument("--sample", type=int, default=5)
    inspect.add_argument("--count", action="store_true", help="Count all rows")
    _add_column_options(inspect)
    inspect.set_defaults(func=cmd_inspect_data)

    features = subparsers.add_parser("prepare-features", help="Build a feature bundle")
    _add_common(features)
    features.add_argument("--pairs", type=Path, required=True)
    features.add_argument("--out", type=Path, required=True)
    features.add_argument("--backend", choices=["esm2", "hash"], default=None)
    features.add_argument("--limit", type=int, default=None)
    features.add_argument("--sample-negatives", action="store_true")
    features.add_argument("--overwrite", action="store_true")
    _add_column_options(features)
    features.set_defaults(func=cmd_prepare_features)

    splits = subparsers.add_parser("make-splits", help="Create train/validation/test splits")
    _add_common(splits)
    splits.add_argument("--pairs", type=Path, required=True)
    splits.add_argument("--out", type=Path, required=True)
    splits.add_argument("--protocol", default=None, choices=["paper", "legacy", "group", "server_historical"])
    splits.add_argument("--n-folds", type=int, default=None)
    splits.add_argument("--test-fraction", type=float, default=None)
    splits.add_argument("--group-by", default=None, choices=["union_find", "virus_species"])
    splits.add_argument("--strict-leakage", action="store_true")
    splits.add_argument("--sample-negatives", action="store_true")
    splits.add_argument("--limit", type=int, default=None, help="Use only the first N rows")
    _add_column_options(splits)
    splits.set_defaults(func=cmd_make_splits)

    pretrain = subparsers.add_parser("train-pretrain", help="Cross-modal pretraining")
    _add_common(pretrain)
    _add_training_options(pretrain)
    pretrain.set_defaults(func=cmd_train_pretrain)

    classifier = subparsers.add_parser("train-classifier", help="Fine-tune the classifier")
    _add_common(classifier)
    _add_training_options(classifier)
    classifier.add_argument("--pretrain-checkpoint", type=Path, default=None)
    classifier.set_defaults(func=cmd_train_classifier)

    evaluate = subparsers.add_parser("evaluate", help="Evaluate a checkpoint")
    _add_common(evaluate)
    evaluate.add_argument("--checkpoint", type=Path, help="One checkpoint for a specified fold")
    evaluate.add_argument("--checkpoint-map", type=Path, help="JSON mapping fold indices to checkpoint paths")
    evaluate.add_argument("--bundle", type=Path, required=True)
    evaluate.add_argument("--splits", type=Path, required=True)
    evaluate.add_argument("--fold", default="0", help="Fold index or 'all'")
    evaluate.add_argument("--split", default=None, choices=["train", "val", "test"])
    evaluate.add_argument("--out", type=Path, default=None)
    evaluate.set_defaults(func=cmd_evaluate)

    predict = subparsers.add_parser("predict", help="Predict probabilities for new pairs")
    _add_common(predict)
    predict.add_argument("--checkpoint", type=Path, default=None)
    predict.add_argument("--esm-dir", type=Path)
    predict.add_argument("--projection", type=Path)
    predict.add_argument("--cache", type=Path)
    predict.add_argument("--pairs", type=Path, default=None)
    predict.add_argument("--bundle", type=Path, default=None)
    predict.add_argument("--splits", type=Path, default=None)
    predict.add_argument("--fold", type=int, default=0)
    predict.add_argument("--split", default="test", choices=["train", "val", "test"])
    predict.add_argument("--limit", type=int, default=None)
    predict.add_argument("--out", type=Path, default=Path("predictions.csv"))
    predict.add_argument("--allow-backend-mismatch", action="store_true")
    _add_reference_preparation_options(predict)
    _add_column_options(predict)
    predict.set_defaults(func=cmd_predict)

    for name, help_text in (
        ("retrieve", "Retrieve nearest reference proteins"),
        ("generate", "Retrieval-based candidate generation (no decoder weights released)"),
    ):
        command = subparsers.add_parser(name, help=help_text)
        _add_common(command)
        command.add_argument("--pairs", type=Path, required=True)
        command.add_argument("--checkpoint", type=Path, default=None,
                             help="Use checkpoint encoding settings and recorded reference indices")
        reference_inputs = command.add_mutually_exclusive_group()
        reference_inputs.add_argument("--reference", type=Path, default=None, help="Reference FASTA")
        reference_inputs.add_argument("--index", type=Path, default=None, help="Prebuilt reference index")
        reference_inputs.add_argument("--reference-source", choices=["human", "viral"],
                                      help="Automatically download, prepare and index a full public gallery")
        command.add_argument("--reference-dir", type=Path, default=None,
                             help="Cache root for public references (default: references.dir or data/reference)")
        command.add_argument("--index-out", type=Path, default=None)
        command.add_argument("--target", default=None, choices=["human", "viral"],
                             help="Side to search (default: reference-source, otherwise viral)")
        command.add_argument("--top-k", type=int, default=None)
        command.add_argument("--limit", type=int, default=None)
        command.add_argument("--out", type=Path, default=Path("retrieval.json"))
        command.add_argument("--overwrite", action="store_true")
        _add_column_options(command)
        command.set_defaults(func=cmd_retrieve if name == "retrieve" else cmd_generate)

    export = subparsers.add_parser("export-legacy", help="Convert a released pickle checkpoint")
    _add_common(export)
    export.add_argument("--checkpoint", type=Path, required=True)
    export.add_argument("--out", type=Path, default=Path("runs/legacy_export"))
    export.add_argument("--projection", type=Path)
    export.add_argument("--cache", type=Path)
    export.add_argument("--esm-dir", type=Path)
    export.add_argument("--allow-unsafe-pickle", action="store_true")
    export.add_argument("--project-root", type=Path, default=None, help=argparse.SUPPRESS)
    export.set_defaults(func=cmd_export_legacy)

    doctor = subparsers.add_parser("doctor", help="Check data, weights and dependencies")
    _add_common(doctor)
    doctor.add_argument("--write", type=Path, default=None)
    doctor.set_defaults(func=cmd_doctor)

    config_show = subparsers.add_parser("config", help="Inspect the resolved configuration")
    _add_common(config_show)
    config_show.add_argument("action", choices=["show"])
    config_show.set_defaults(func=cmd_config_show)

    verify = subparsers.add_parser("verify-resources", help="Verify resource inventory and SHA-256 checksums")
    _add_common(verify)
    verify.add_argument("--require-release", action="store_true")
    verify.set_defaults(func=cmd_verify_resources)

    reproduce = subparsers.add_parser("reproduce", help="Audit, encode, split, pretrain and classify every fold")
    _add_common(reproduce)
    reproduce.add_argument("--pairs", type=Path, required=True)
    reproduce.add_argument("--out", type=Path, required=True)
    reproduce.add_argument("--limit", type=int)
    reproduce.add_argument("--epochs", type=int, help="Override both training stages for smoke tests")
    _add_reference_preparation_options(reproduce)
    reproduce.set_defaults(func=cmd_reproduce)

    recompute = subparsers.add_parser("recompute-metrics", help="Recompute metrics from saved sample probabilities")
    _add_common(recompute)
    recompute.add_argument("--predictions", type=Path, required=True)
    recompute.add_argument("--out", type=Path, required=True)
    recompute.set_defaults(func=cmd_recompute_metrics)
    return parser


def _add_column_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--human-col", default=None)
    parser.add_argument("--viral-col", default=None)
    parser.add_argument("--label-col", default=None)
    parser.add_argument("--pair-id-col", default=None)
    parser.add_argument("--human-id-col", default=None)
    parser.add_argument("--virus-id-col", default=None)
    parser.add_argument("--species-col", default=None)


def _add_training_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--run-name", default=None)
    _add_reference_preparation_options(parser)


def _add_reference_preparation_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--reference-source", choices=["human", "viral", "both"],
                        help="Prepare public reference indices alongside this workflow")
    parser.add_argument("--reference-dir", type=Path, default=None,
                        help="Public reference cache root (default: references.dir or data/reference)")
    parser.add_argument("--reference-overwrite", action="store_true", default=None,
                        help="Rebuild reference indices from verified downloaded sequences")


def _reference_preparation_config(config, args):
    settings = config.setdefault("references", {"sources": [], "dir": "data/reference", "overwrite": False})
    source = getattr(args, "reference_source", None)
    if source:
        settings["sources"] = ["human", "viral"] if source == "both" else [source]
    if getattr(args, "reference_dir", None) is not None:
        settings["dir"] = str(args.reference_dir)
    if getattr(args, "reference_overwrite", None) is not None:
        settings["overwrite"] = args.reference_overwrite
    return config


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _config(args: argparse.Namespace, *, extra: Sequence[str] | None = None) -> dict[str, Any]:
    config = load_config(getattr(args, "config", None), list(getattr(args, "overrides", []) or []) + list(extra or []))
    return _apply_common(config, args)


def _apply_common(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    device = getattr(args, "device", None)
    if device:
        config["run"]["device"] = device
    seed = getattr(args, "seed", None)
    if seed is not None:
        config["run"]["seed"] = int(seed)
        config["data"]["splits"]["seed"] = int(seed)
        config["data"]["negatives"]["seed"] = int(seed)
    from .config import load_yaml

    resources = config.setdefault("resources", {})
    if getattr(args, "resources_dir", None):
        resources["dir"] = str(args.resources_dir)
    if not resources.get("dir") and Path("data/manifest.json").is_file():
        resources["dir"] = "data"
    paths = resources.setdefault("paths", {})
    for key, attr in (("esm", "esm_dir"), ("projection", "projection"), ("cache", "cache")):
        value = getattr(args, attr, None)
        if value is not None:
            paths[key] = str(value)
    # Defaults must not take precedence over bundle locations. Only explicit
    # config values / --set overrides count as user-specified paths.
    raw = load_yaml(args.config) if getattr(args, "config", None) else {}
    raw = apply_overrides(raw, getattr(args, "overrides", []) or [])
    explicit_model = raw.get("data", {}).get("esm", {}).get("model")
    from .data.esm import DEFAULT_MODEL_ALIASES
    from .resources.bundle import DEFAULTS, safe_path

    if paths.get("esm"):
        config["data"]["esm"]["model"] = paths["esm"]
    elif explicit_model and explicit_model not in DEFAULT_MODEL_ALIASES:
        config["data"]["esm"]["model"] = explicit_model
    elif resources.get("dir") and config["data"]["esm"]["backend"] != "hash":
        # Resolve paths here; loading and integrity checks belong to the encoder.
        # Inspecting data, config, splits or resource health requires no model.
        config["data"]["esm"]["model"] = str(safe_path(resources["dir"], DEFAULTS["esm"]))
    return config


def _config_from_payload(payload: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    """Use the checkpoint's training config, with prediction-time overrides.

    Architecture settings always come from the checkpoint, because the model
    was already built from them. A user-supplied --config may adjust the data
    adapter, evaluation and prediction sections.
    """
    import copy

    config = payload.get("config")
    if not isinstance(config, dict):
        raise CheckpointError("Checkpoint does not contain its training configuration")
    config = copy.deepcopy(config)
    if getattr(args, "config", None):
        from .config import load_yaml

        resolved = load_yaml(args.config)
        for section in ("data", "eval", "predict", "retrieval", "run", "resources", "references"):
            config[section] = deep_merge(config.get(section, {}), resolved.get(section, {}))
    config = apply_overrides(config, list(getattr(args, "overrides", []) or []))
    if payload.get("model_name") == "server_tmonet_v1":
        # The checkpoint's numerical encoding contract cannot be overridden.
        config["data"]["esm"].update(backend="server_esm2", pooling="attention_mean", max_length=SERVER_MAX_LENGTH)
        config["data"]["aaindex"].update(table=None, max_residues=100, dim=1200)
        config.setdefault("resources", {})["fingerprints"] = payload["resource_fingerprints"]
    return _apply_common(config, args)


def _columns(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, str | None]:
    columns = dict(config["data"].get("columns") or {})
    mapping = {
        "human_seq": "human_col",
        "viral_seq": "viral_col",
        "label": "label_col",
        "pair_id": "pair_id_col",
        "human_id": "human_id_col",
        "virus_id": "virus_id_col",
        "virus_species": "species_col",
    }
    for key, attribute in mapping.items():
        value = getattr(args, attribute, None)
        if value:
            columns[key] = value
    return columns


def _read_pairs(
    path: Path,
    args: argparse.Namespace,
    config: dict[str, Any],
    *,
    require_label: bool,
    limit: int | None = None,
    training: bool = False,
):
    reader = read_training_pairs if training else read_pair_table
    options = {"protocol": config["data"]["splits"]["protocol"]} if training else {"strict": True, "deduplicate": False}
    table = reader(
        path,
        columns=_columns(args, config),
        limit=limit,
        require_label=require_label,
        **options,
    )
    if table.errors:
        get_logger().warning(
            "Skipped %d invalid row(s); first: %s", len(table.errors), table.errors[0]
        )
    return table


def _maybe_sample_negatives(
    records: Sequence[PairRecord], config: dict[str, Any], enabled: bool
) -> tuple[list[PairRecord], dict[str, Any] | None]:
    if not enabled:
        return list(records), None
    settings = config["data"]["negatives"]
    records, report = ensure_negatives(
        records,
        ratio=float(settings["ratio"]),
        seed=int(settings["seed"]),
        strategy=str(settings.get("strategy", "random")),
    )
    return records, report.to_dict()


def _feature_spec(bundle: FeatureBundle) -> dict[str, Any]:
    manifest = bundle.manifest
    return {
        "dims": manifest["dims"],
        "sequence_backend": manifest["sequence_backend"],
        "aaindex": manifest["aaindex"],
        "records_fingerprint": manifest.get("records_fingerprint"),
    }


def _check_bundle_matches_splits(bundle: FeatureBundle, assignment) -> None:
    expected = assignment.manifest.get("records_fingerprint")
    actual = bundle.fingerprint
    if expected and actual and expected != actual:
        raise DataFormatError(
            "The feature bundle and the split file describe different pair tables "
            f"({actual} vs {expected}). Rebuild one of them."
        )


def _loaders(bundle: FeatureBundle, fold: dict[str, Any], config: dict[str, Any], stage: str):
    settings = config[stage]
    batch_size = int(settings["batch_size"])
    seed = int(config["run"]["seed"])
    num_workers = int(config["run"].get("num_workers", 0))
    train_view = bundle.view(fold["train"])
    train_loader = build_dataloader(
        train_view,
        batch_size=batch_size,
        shuffle=True,
        drop_last=bool(settings.get("drop_last_train", True)) and len(train_view) > batch_size,
        num_workers=num_workers,
        seed=seed,
    )
    val_loader = build_dataloader(
        bundle.view(fold["val"]),
        batch_size=max(32, batch_size),
        shuffle=False,
        num_workers=num_workers,
        seed=seed,
    )
    test_loader = build_dataloader(
        bundle.view(fold["test"]),
        batch_size=max(32, batch_size),
        shuffle=False,
        num_workers=num_workers,
        seed=seed,
    )
    return train_loader, val_loader, test_loader


class _FixedRunDirectory:
    """Minimal RunDirectory-compatible object for the self-contained demo."""

    def __init__(self, root: Path):
        self.root = root
        self.checkpoints = root / "checkpoints"
        self.checkpoints.mkdir(parents=True, exist_ok=True)

    def checkpoint(self, name: str) -> Path:
        return self.checkpoints / name


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def cmd_demo(args: argparse.Namespace) -> int:
    config = _config(args)
    config["data"]["esm"]["backend"] = args.backend
    if args.backend == "hash" or config["data"]["esm"].get("model") == "hash":
        config["data"]["esm"]["model"] = args.backend
    if args.backend == "hash":
        config["run"]["device"] = "cpu"
    device = resolve_device(config["run"]["device"])
    seed_everything(int(config["run"]["seed"]), deterministic=config["run"]["deterministic"])

    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    setup_logging(getattr(args, "log_level", "INFO"), output / "run.log")
    logger = get_logger()
    logger.info("DeepHVI demo | output=%s backend=%s", output, args.backend)
    if args.backend == "hash":
        logger.warning(
            "backend=hash uses deterministic k-mer feature hashing, not ESM-2. "
            "Demo metrics are pipeline checks, not biological results."
        )

    records = generate_synthetic_records(
        n_pairs=args.n_pairs, positive_fraction=args.positive_fraction, seed=int(config["run"]["seed"])
    )
    pairs_path = write_pairs_csv(records, output / "data" / "pairs.csv")
    logger.info("Wrote %d synthetic pairs to %s", len(records), pairs_path)

    assignment = make_splits(
        records,
        protocol="paper",
        n_folds=int(config["data"]["splits"]["n_folds"]),
        test_fraction=float(config["data"]["splits"]["test_fraction"]),
        seed=int(config["data"]["splits"]["seed"]),
    )
    save_splits(assignment, output / "splits")

    bundle = prepare_feature_bundle(
        records, config=config, output_dir=output / "features", device=str(device), overwrite=True
    )
    spec = _feature_spec(bundle)
    fold = assignment.fold(0)
    train_loader, val_loader, test_loader = _loaders(bundle, fold, config, "train_pretrain")
    model = build_model(config).to(device)
    run_dir = _FixedRunDirectory(output)

    pretrain_result = train_pretrain(
        model,
        train_loader,
        val_loader,
        config=config,
        device=device,
        feature_spec=spec,
        split_fingerprint=assignment.manifest.get("records_fingerprint"),
        run_dir=run_dir,
        epochs=args.epochs if args.epochs is not None else int(config["train_pretrain"]["epochs"]),
    )
    logger.info("Pretraining finished; best epoch=%s", pretrain_result.best_epoch)

    train_loader, val_loader, test_loader = _loaders(bundle, fold, config, "train_classifier")
    model = build_model(config).to(device)
    classifier_result = train_classifier(
        model,
        train_loader,
        val_loader,
        config=config,
        device=device,
        feature_spec=spec,
        split_fingerprint=assignment.manifest.get("records_fingerprint"),
        run_dir=run_dir,
        epochs=args.epochs if args.epochs is not None else int(config["train_classifier"]["epochs"]),
        pretrain_checkpoint=pretrain_result.best_checkpoint,
    )
    logger.info("Classifier training finished; best epoch=%s", classifier_result.best_epoch)

    checkpoint_path = classifier_result.best_checkpoint or str(run_dir.checkpoint("last.pt"))
    model, payload = load_model_from_checkpoint(checkpoint_path, device=device)
    test_result = evaluate_loader(model, test_loader, device, config=config)
    write_predictions(test_result, output / "predictions" / "test_predictions.csv", threshold=config["eval"]["threshold"])
    write_metrics({0: test_result["metrics"]}, output / "metrics" / "test_metrics.json")
    logger.info("Test metrics: %s", format_metrics(test_result["metrics"]))

    test_records = [records[index] for index in fold["test"][: min(8, len(fold["test"]))]]
    prediction_frame = predict_records(
        model, test_records, config=config, device=device, batch_size=int(config["predict"]["batch_size"])
    )
    prediction_frame.to_csv(output / "predictions" / "new_pairs.csv", index=False)

    reference_sequences = list(dict.fromkeys(record.viral_seq for record in records))
    reference_ids = [
        next(record.virus_id or record.pair_id for record in records if record.viral_seq == sequence)
        for sequence in reference_sequences
    ]
    index = build_index_from_sequences(
        reference_sequences,
        output / "retrieval" / "reference_index",
        config=config,
        ids=reference_ids,
        device=str(device),
        overwrite=True,
    )
    hits = retrieve_for_records(
        test_records[:5],
        index,
        target="viral",
        config=config,
        device=str(device),
        top_k=int(config["retrieval"]["top_k"]),
    )
    write_json(
        output / "retrieval" / "candidates.json",
        {
            "note": "Retrieval-based candidate generation; decoder weights were not released.",
            "queries": [
                {"pair_id": record.pair_id, "hits": hit_list}
                for record, hit_list in zip(test_records[:5], hits, strict=False)
            ],
        },
    )

    summary = {
        "backend": args.backend,
        "n_pairs": len(records),
        "device": str(device),
        "pretrain": pretrain_result.to_dict(),
        "classifier": classifier_result.to_dict(),
        "test_metrics": test_result["metrics"],
        "confusion_matrix": test_result["confusion_matrix"],
        "paths": {
            "pairs": str(pairs_path),
            "features": str(bundle.path),
            "splits": str(output / "splits"),
            "checkpoint": checkpoint_path,
            "predictions": str(output / "predictions"),
            "metrics": str(output / "metrics"),
            "retrieval": str(output / "retrieval"),
        },
        "warning": "Synthetic data and hash encoder: not a reproduction of paper results."
        if args.backend == "hash"
        else "Synthetic data: not a reproduction of paper results.",
    }
    write_json(output / "demo_summary.json", summary)
    print(json.dumps({"test_metrics": test_result["metrics"], "output": str(output)}, indent=2))
    logger.info("Demo complete. Summary: %s", output / "demo_summary.json")
    return 0


def cmd_prepare_data(args: argparse.Namespace) -> int:
    report = prepare_dataset(args.archive, args.dest, overwrite=args.overwrite)
    print(json.dumps(report, indent=2))
    if not report["tables"]:
        get_logger().warning("No CSV files were found in the extracted archive.")
    return 0


def cmd_inspect_data(args: argparse.Namespace) -> int:
    config = _config(args)
    report = inspect_table(
        args.input, columns=_columns(args, config), sample_rows=args.sample
    )
    if args.count and not report.get("error"):
        from .data.legacy import count_rows

        try:
            report["n_rows"] = count_rows(args.input)
        except Exception as exc:  # noqa: BLE001
            report["count_error"] = str(exc)
    print(json.dumps(report, indent=2, default=str))
    return 0


def cmd_prepare_features(args: argparse.Namespace) -> int:
    config = _config(args, extra=([f"data.esm.backend={args.backend}"] if args.backend else []))
    if args.backend == "hash":
        config["data"]["esm"]["model"] = "hash"
    device = resolve_device(config["run"]["device"])
    table = _read_pairs(args.pairs, args, config, require_label=args.sample_negatives, limit=args.limit, training=True)
    records, report = _maybe_sample_negatives(table.records, config, args.sample_negatives)
    bundle = prepare_feature_bundle(
        records,
        config=config,
        output_dir=args.out,
        device=str(device),
        overwrite=args.overwrite,
    )
    summary = dict(bundle.manifest)
    summary["preprocessing"] = table.duplicate_report
    if report is not None:
        summary["negative_sampling"] = report
    print(json.dumps(summary, indent=2, default=str))
    return 0


def cmd_make_splits(args: argparse.Namespace) -> int:
    config = _config(args)
    overrides = {
        "protocol": args.protocol,
        "n_folds": args.n_folds,
        "test_fraction": args.test_fraction,
        "group_by": args.group_by,
    }
    for key, value in overrides.items():
        if value is not None:
            config["data"]["splits"][key] = value
    table = _read_pairs(args.pairs, args, config, require_label=True, limit=args.limit, training=True)
    records, report = _maybe_sample_negatives(table.records, config, args.sample_negatives)
    settings = config["data"]["splits"]
    assignment = make_splits(
        records,
        protocol=settings["protocol"],
        n_folds=int(settings["n_folds"]),
        test_fraction=float(settings["test_fraction"]),
        seed=int(settings["seed"]),
        legacy_fold_index=int(settings["legacy_fold_index"]),
        legacy_n_folds=int(settings["legacy_n_folds"]),
        group_by=settings["group_by"],
        strict_leakage=args.strict_leakage,
    )
    assignment.manifest["preprocessing"] = table.duplicate_report
    save_splits(assignment, args.out)
    summary = dict(assignment.manifest)
    summary["negative_sampling"] = report
    print(json.dumps(summary, indent=2, default=str))
    return 0


def _training_setup(args: argparse.Namespace):
    config = _reference_preparation_config(_config(args), args)
    device = resolve_device(config["run"]["device"])
    seed_everything(int(config["run"]["seed"]), deterministic=config["run"]["deterministic"])
    bundle = FeatureBundle.load(args.bundle)
    assignment = load_splits(args.splits)
    _check_bundle_matches_splits(bundle, assignment)
    backend = verify_bundle_encoding(bundle, config, device=str(device))
    from .training.references import prepare_training_references, record_training_references

    initialization = args.resume or getattr(args, "pretrain_checkpoint", None)
    if not initialization and args.command == "train-classifier":
        initialization = config["train_classifier"].get("pretrain_checkpoint")
    inherited = load_checkpoint(initialization).get("reference_libraries", {}) if initialization else None
    if args.resume and inherited and config.get("references", {}).get("overwrite"):
        raise CheckpointError("Resume keeps reference indices pinned; omit --reference-overwrite")
    references = prepare_training_references(config, device=str(device), backend=backend,
                                             inherited=inherited)
    if args.resume and inherited and references != inherited:
        from .inference.reference_pipeline import verify_recorded_reference

        if set(references) != set(inherited):
            raise CheckpointError("Resume reference sources differ from the saved checkpoint")
        for side, record in references.items():
            verify_recorded_reference(ReferenceIndex.load(record["index"]), inherited[side])
    del backend
    # Constructing a verification encoder may consume RNGs; model initialization
    # must depend on the run seed, not whether cached features were checked.
    seed_everything(int(config["run"]["seed"]), deterministic=config["run"]["deterministic"])
    fold = assignment.fold(args.fold)
    config["run"]["training_fold"] = args.fold
    config["run"]["split_fingerprint"] = assignment.fingerprint
    if args.run_name:
        config["run"]["name"] = args.run_name
    run_dir = RunDirectory(
        config["run"]["output_dir"], name=config["run"]["name"], config=config_for_json(config)
    )
    record_training_references(config, references, run_dir.root)
    (run_dir.root / "resolved_config.yaml").write_text(yaml.safe_dump(config_for_json(config), sort_keys=True))
    setup_logging(getattr(args, "log_level", "INFO"), run_dir.root / "run.log")
    return config, device, bundle, assignment, fold, run_dir


def cmd_train_pretrain(args: argparse.Namespace) -> int:
    config, device, bundle, assignment, fold, run_dir = _training_setup(args)
    train_loader, val_loader, _ = _loaders(bundle, fold, config, "train_pretrain")
    model = build_model(config).to(device)
    result = train_pretrain(
        model,
        train_loader,
        val_loader,
        config=config,
        device=device,
        feature_spec=_feature_spec(bundle),
        split_fingerprint=assignment.manifest.get("records_fingerprint"),
        run_dir=run_dir,
        epochs=args.epochs,
        resume_from=str(args.resume) if args.resume else None,
    )
    write_json(run_dir.metric("training.json"), result.to_dict())
    print(json.dumps({"best_epoch": result.best_epoch, "best_checkpoint": result.best_checkpoint}, indent=2))
    return 0


def cmd_train_classifier(args: argparse.Namespace) -> int:
    config, device, bundle, assignment, fold, run_dir = _training_setup(args)
    train_loader, val_loader, _ = _loaders(bundle, fold, config, "train_classifier")
    model = build_model(config).to(device)
    result = train_classifier(
        model,
        train_loader,
        val_loader,
        config=config,
        device=device,
        feature_spec=_feature_spec(bundle),
        split_fingerprint=assignment.manifest.get("records_fingerprint"),
        run_dir=run_dir,
        epochs=args.epochs,
        resume_from=str(args.resume) if args.resume else None,
        pretrain_checkpoint=str(args.pretrain_checkpoint) if args.pretrain_checkpoint else config["train_classifier"].get("pretrain_checkpoint"),
    )
    write_json(run_dir.metric("training.json"), result.to_dict())
    print(json.dumps({"best_epoch": result.best_epoch, "best_checkpoint": result.best_checkpoint}, indent=2))
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    from .evaluation.folds import checkpoint_paths, validate_fold_checkpoint

    assignment = load_splits(args.splits)
    paths = checkpoint_paths(args.checkpoint, args.checkpoint_map, args.fold, assignment.n_folds)
    bundle = FeatureBundle.load(args.bundle)
    _check_bundle_matches_splits(bundle, assignment)
    folds = list(paths)
    config = _config(args)
    split_name = args.split or config["eval"]["split"]
    per_fold: dict[int, dict[str, Any]] = {}
    results: dict[int, dict[str, Any]] = {}
    for fold_index in folds:
        model, payload = load_model_from_checkpoint(str(paths[fold_index]), device="cpu")
        validate_fold_checkpoint(payload, fold_index, assignment, require_metadata=len(paths) > 1)
        config = _config_from_payload(payload, args)
        device = resolve_device(config["run"]["device"])
        model.to(device)
        verify_feature_spec(model, bundle.manifest)
        from .inference.predict import check_aaindex_matches, check_feature_backend

        check_feature_backend(payload, bundle.manifest["sequence_backend"])
        check_aaindex_matches(payload, bundle.manifest["aaindex"])
        fold = assignment.fold(fold_index)
        indices = fold[split_name]
        loader = build_dataloader(
            bundle.view(indices), batch_size=max(32, int(config["predict"]["batch_size"])), shuffle=False
        )
        result = evaluate_loader(model, loader, device, config=config)
        per_fold[fold_index] = result["metrics"]
        results[fold_index] = result

    output = Path(args.out) if args.out else next(iter(paths.values())).resolve().parent.parent
    metrics_dir = output / "metrics"
    predictions_dir = output / "predictions"
    write_metrics(per_fold, metrics_dir / f"{split_name}_metrics.json")
    aggregate = aggregate_metrics([dict(metrics) for metrics in per_fold.values()])
    markdown = write_markdown_report(
        metrics_dir / f"{split_name}_report.md",
        title=f"DeepHVI evaluation ({split_name})",
        per_fold=per_fold,
        extra_lines=[
            f"- Checkpoint: `{args.checkpoint}`",
            f"- Split file: `{args.splits}`",
            f"- Feature bundle: `{args.bundle}`",
            f"- Protocol: {assignment.protocol}",
        ],
    )
    for fold_index, result in results.items():
        write_predictions(result, predictions_dir / f"{split_name}_fold{fold_index}.csv", threshold=config["eval"]["threshold"])
    write_json(metrics_dir / f"{split_name}_aggregate.json", aggregate)
    print(json.dumps({"metrics": per_fold, "aggregate": aggregate}, indent=2, default=float))
    get_logger().info("Wrote %s and %s", metrics_dir, markdown)
    return 0


def cmd_predict(args: argparse.Namespace) -> int:
    from .resources.bundle import resolve_resource

    initial = _config(args)
    args.checkpoint = resolve_resource("checkpoint", explicit=args.checkpoint,
        configured=initial.get("predict", {}).get("checkpoint"), resources_dir=initial.get("resources", {}).get("dir"))
    model, payload = load_model_for_prediction(args.checkpoint, device="cpu")
    config = _config_from_payload(payload, args)
    device = resolve_device(config["run"]["device"])
    model.to(device)
    threshold = float(config["eval"]["threshold"])

    if args.bundle:
        if not args.splits:
            raise ValueError("--bundle requires --splits")
        bundle = FeatureBundle.load(args.bundle)
        verify_feature_spec(model, bundle.manifest)
        from .inference.predict import check_aaindex_matches, check_feature_backend

        check_feature_backend(payload, bundle.manifest["sequence_backend"])
        check_aaindex_matches(payload, bundle.manifest["aaindex"])
        assignment = load_splits(args.splits)
        _check_bundle_matches_splits(bundle, assignment)
        indices = assignment.fold(args.fold)[args.split]
        frame = predict_bundle(
            model,
            bundle,
            config=config,
            device=device,
            indices=indices,
            batch_size=int(config["predict"]["batch_size"]),
        )
    else:
        if not args.pairs:
            raise ValueError("Provide --pairs or --bundle/--splits")
        from .data.sequence_backend import build_backend

        backend = build_backend(config, device=str(device))
        from .data.features import aaindex_spec
        from .inference.predict import check_aaindex_matches

        check_aaindex_matches(payload, aaindex_spec(config))
        if not args.allow_backend_mismatch:
            check_backend_matches(payload, backend)
        table = _read_pairs(args.pairs, args, config, require_label=False, limit=args.limit)
        frame = predict_records(
            model,
            table.records,
            config=config,
            device=device,
            backend=backend,
            batch_size=int(config["predict"]["batch_size"]),
        )
        frame["prediction"] = (frame["probability"] >= threshold).astype(int)

    preparation = _reference_preparation_config(initial, args).get("references", {})
    summary = {"n_pairs": len(frame), "output": str(args.out)}
    if preparation.get("sources"):
        from .training.references import prepare_training_references

        config["references"] = preparation
        references = prepare_training_references(config, device=str(device))
        reference_path = args.out.with_name(args.out.name + ".references.json")
        write_json(reference_path, references)
        summary["references"] = str(reference_path)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    print(json.dumps(summary, indent=2))
    return 0


def _retrieve_command(args: argparse.Namespace, *, generate: bool) -> int:
    payload = load_checkpoint(args.checkpoint) if args.checkpoint else None
    config = _config_from_payload(payload, args) if payload else _config(args)
    device = resolve_device(config["run"]["device"])
    table = _read_pairs(args.pairs, args, config, require_label=False, limit=args.limit)
    top_k = int(args.top_k if args.top_k is not None else config["retrieval"]["top_k"])
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    target = args.target or args.reference_source or "viral"
    if args.reference_source and args.reference_source != target:
        raise ValueError("--reference-source must match --target")
    recorded = (payload or {}).get("reference_libraries", {})
    selected_record = recorded.get(target)
    source = args.reference_source
    if (not source and not (args.index or args.reference or selected_record)
            and target in config.get("references", {}).get("sources", [])):
        source = target
    if not (args.index or args.reference or source or selected_record):
        raise ValueError("Provide --reference FASTA, --index directory or --reference-source human|viral")
    if args.index and args.index_out:
        raise ValueError("--index-out cannot be used with --index")

    from .data.sequence_backend import build_backend

    backend = build_backend(config, device=str(device))
    if payload:
        check_backend_matches(payload, backend)

    if source:
        from .inference.reference_pipeline import ensure_public_reference

        index = ensure_public_reference(
            source, args.reference_dir or Path(config.get("references", {}).get("dir", "data/reference")),
            config=config, device=str(device),
            backend=backend, index_dir=args.index_out, overwrite=args.overwrite,
        )
    elif args.index or (selected_record and not args.reference):
        index = ReferenceIndex.load(args.index or selected_record["index"])
        trained_backend = index.spec.get("backend")
        if trained_backend and trained_backend != backend.name:
            raise RetrievalError(
                f"Reference index was built with backend={trained_backend!r}, "
                f"but the current backend is {backend.name!r}."
            )
    else:
        index_dir = args.index_out or (Path(args.out).parent / "reference_index")
        index = build_reference_index(
            read_fasta(args.reference),
            index_dir,
            config=config,
            device=str(device),
            backend=backend,
            overwrite=args.overwrite,
        )

    reference = index.manifest.get("reference")
    if reference and reference["source"] != target:
        raise RetrievalError("Public reference index source must match --target")
    if selected_record and not args.reference:
        from .inference.reference_pipeline import verify_recorded_reference

        verify_recorded_reference(index, selected_record)

    hits = retrieve_for_records(
        table.records,
        index,
        target=target,
        config=config,
        device=str(device),
        backend=backend,
        top_k=top_k,
    )
    rows = []
    for record, hit_list in zip(table.records, hits, strict=False):
        query = record.human_seq if target == "human" else record.viral_seq
        for hit in hit_list:
            rows.append(
                {
                    "pair_id": record.pair_id,
                    "target": target,
                    "query_sequence": query,
                    "rank": hit["rank"],
                    "hit_id": hit["id"],
                    "score": hit["score"],
                    "hit_description": hit["description"],
                    "hit_sequence": hit["sequence"],
                }
            )
    if Path(args.out).suffix.lower() == ".json":
        write_json(
            args.out,
            {
                "note": "Retrieval-based candidate generation; no decoder weights were released."
                if generate
                else "Cosine retrieval results.",
                "target": target,
                "top_k": top_k,
                "index": str(index.path),
                "results": rows,
                **({"reference": index.manifest["reference"]}
                   if "reference" in index.manifest else {}),
            },
        )
    else:
        import pandas as pd

        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(args.out, index=False)
    print(json.dumps({"n_queries": len(table.records), "top_k": top_k,
                      "index": str(index.path), "output": str(args.out)}, indent=2))
    return 0


def cmd_retrieve(args: argparse.Namespace) -> int:
    return _retrieve_command(args, generate=False)


def cmd_generate(args: argparse.Namespace) -> int:
    return _retrieve_command(args, generate=True)


def cmd_export_legacy(args: argparse.Namespace) -> int:
    report = export_legacy_checkpoint(
        args.checkpoint,
        args.out,
        allow_unsafe_pickle=args.allow_unsafe_pickle,
        project_root=args.project_root,
        projection=args.projection, cache=args.cache, esm_dir=args.esm_dir,
    )
    print(json.dumps(report, indent=2))
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    config = _config(args)
    report = collect_doctor_report(Path.cwd(), resources_dir=config.get("resources", {}).get("dir"))
    for check in report["checks"]:
        print(f"[{check['status']:9s}] tier {check['tier']} {check['label']}: {check['detail']}")
    if args.write:
        path = write_missing_resources(report, args.write)
        print(f"\nWrote {path}")
    return 0


def cmd_config_show(args: argparse.Namespace) -> int:
    config = _config(args)
    print(yaml.safe_dump(config_for_json(config), sort_keys=False))
    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def cmd_verify_resources(args):
    from .resources.bundle import verify_bundle

    config = _config(args)
    root = config.get("resources", {}).get("dir")
    if not root:
        raise ValueError("Pass --resources-dir to the unpacked resource bundle")
    print(json.dumps(verify_bundle(root, require_release=args.require_release), indent=2))
    return 0


def cmd_reproduce(args):
    from .reproduction import reproduce

    config = _reference_preparation_config(_config(args), args)
    result = reproduce(args.pairs, args.out, config=config, limit=args.limit, epochs=args.epochs)
    print(json.dumps(result, indent=2))
    return 0


def cmd_recompute_metrics(args):
    from .evaluation.metrics import metrics_from_csv

    config = _config(args)
    result = metrics_from_csv(args.predictions, threshold=config["eval"]["threshold"], average=config["eval"]["average"])
    write_json(args.out, result)
    print(json.dumps(result, indent=2))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    setup_logging(getattr(args, "log_level", "INFO"))
    try:
        return int(args.func(args) or 0)
    except KeyboardInterrupt:
        return 130
    except (
        AAindexError,
        ArchiveError,
        CheckpointError,
        DataFormatError,
        ESMBackendError,
        FeatureBundleError,
        LeakageError,
        LegacyExportError,
        PairTableError,
        RetrievalError,
        ReferenceDataError,
        SplitError,
        FileNotFoundError,
        ValueError,
    ) as exc:
        get_logger().error("%s", exc)
        return 2


if __name__ == "__main__":
    sys.exit(main())
