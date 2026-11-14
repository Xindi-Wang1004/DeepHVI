"""Audited end-to-end runner. CPU smoke runs explicitly limit samples and epochs."""
from itertools import zip_longest
from pathlib import Path

import yaml

from .data.audit import audit_csv
from .data.features import prepare_feature_bundle
from .data.legacy import read_training_pairs, write_pairs_csv
from .data.splits import make_splits, save_splits
from .evaluation.metrics import metrics_from_csv
from .io import write_json
from .models.registry import build_model, load_model_from_checkpoint
from .paths import environment_info
from .seeding import resolve_device, seed_everything
from .training.trainer import train_classifier, train_pretrain


def reproduce(pairs, output, *, config, limit=None, epochs=None):
    from .cli import _feature_spec, _FixedRunDirectory, _loaders
    from .evaluation.evaluate import evaluate_loader
    from .evaluation.report import write_predictions

    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Refusing to reuse nonempty reproduction directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    report = audit_csv(pairs, columns=config["data"].get("columns"))
    write_json(output / "data_audit.json", report)
    if report["invalid_rows"] or report["conflicting_label_additions"]:
        raise ValueError("Dataset contains invalid or conflicting rows; see data_audit.json")
    table = read_training_pairs(pairs, protocol=config["data"]["splits"]["protocol"], require_label=True,
                                columns=config["data"].get("columns"))
    report["preprocessing"] = table.duplicate_report
    write_json(output / "data_audit.json", report)
    records = table.records
    if limit is not None:
        if limit < 12:
            raise ValueError("Smoke-test --limit must be at least 12 for three-fold splitting")
        # Interleave labels so a positive-first legacy table still exercises both classes.
        groups = [[r for r in records if r.label == label] for label in (0, 1)]
        records = [record for row in zip_longest(*groups) for record in row if record is not None][:limit]
    if len(records) < 12 or len({r.label for r in records}) != 2:
        raise ValueError("Reproduction requires at least 12 records and both labels")
    device = resolve_device(config["run"]["device"])
    seed_everything(config["run"]["seed"], deterministic=config["run"]["deterministic"])
    write_pairs_csv(records, output / "pairs.csv")
    settings = config["data"]["splits"]
    assignment = make_splits(records, protocol=settings["protocol"], n_folds=settings["n_folds"],
                             test_fraction=settings["test_fraction"], seed=settings["seed"],
                             legacy_fold_index=settings["legacy_fold_index"], legacy_n_folds=settings["legacy_n_folds"],
                             group_by=settings["group_by"])
    assignment.manifest["preprocessing"] = table.duplicate_report
    save_splits(assignment, output / "splits")
    config["run"]["split_fingerprint"] = assignment.fingerprint
    (output / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    write_json(output / "environment.json", environment_info())
    bundle = prepare_feature_bundle(records, config=config, output_dir=output / "features", device=str(device))
    from .training.references import prepare_training_references, record_training_references

    references = prepare_training_references(config, device=str(device))
    record_training_references(config, references, output)
    (output / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    paths, summaries = {}, {}
    for index in range(assignment.n_folds):
        config["run"]["training_fold"] = index
        seed_everything(config["run"]["seed"], deterministic=config["run"]["deterministic"])
        model = build_model(config).to(device)
        run = _FixedRunDirectory(output / f"fold{index}")
        train, val, _ = _loaders(bundle, assignment.fold(index), config, "train_pretrain")
        pretrain = train_pretrain(model, train, val, config=config, device=device,
                                  feature_spec=_feature_spec(bundle), run_dir=run, epochs=epochs)
        train, val, test = _loaders(bundle, assignment.fold(index), config, "train_classifier")
        classifier = train_classifier(model, train, val, config=config, device=device,
                                      feature_spec=_feature_spec(bundle), run_dir=run, epochs=epochs,
                                      pretrain_checkpoint=pretrain.best_checkpoint)
        if not classifier.best_checkpoint:
            raise ValueError(f"Fold {index} did not produce a selected checkpoint")
        paths[str(index)] = str(Path(classifier.best_checkpoint).relative_to(output))
        summaries[str(index)] = {"pretrain": pretrain.to_dict(), "classifier": classifier.to_dict()}
    # Every selection uses validation only. Test is accessed after training all folds.
    for index in range(assignment.n_folds):
        model, payload = load_model_from_checkpoint(str(output / paths[str(index)]), device=device)
        from .evaluation.folds import validate_fold_checkpoint

        validate_fold_checkpoint(payload, index, assignment, require_metadata=True)
        _, _, test = _loaders(bundle, assignment.fold(index), config, "train_classifier")
        result = evaluate_loader(model, test, device, config=config)
        destination = output / "predictions" / f"test_fold{index}.csv"
        write_predictions(result, destination, threshold=config["eval"]["threshold"])
        summaries[str(index)]["test"] = metrics_from_csv(destination, threshold=config["eval"]["threshold"], average=config["eval"]["average"])
    write_json(output / "checkpoint_map.json", paths)
    write_json(output / "results.json", summaries)
    summary = {"n_pairs": len(records), "n_folds": assignment.n_folds, "limited_test": limit is not None or epochs is not None,
               "scope": "Code verification only; not paper results" if limit or epochs else "Reproduction run",
               "output": str(output), "checkpoint_map": str(output / "checkpoint_map.json")}
    if references:
        summary["references"] = str(output / "references.json")
    write_json(output / "summary.json", summary)
    return summary
