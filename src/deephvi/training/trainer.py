"""Training loops for pretraining and classifier fine-tuning."""

from __future__ import annotations

import math
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..checkpoints import (
    CheckpointError,
    load_checkpoint,
    load_train_state,
    save_checkpoint,
    save_train_state,
)
from ..data.dataset import move_batch
from ..evaluation.metrics import binary_classification_metrics, format_metrics
from ..logging_utils import get_logger
from ..models.classifier import HVPPIClassifier
from ..models.losses import classification_objective, pretrain_loss
from ..models.registry import parameter_groups, pretrain_parameters
from ..paths import RunDirectory
from ..seeding import seed_everything

MODALITY_KEYS = ("human_esm", "viral_esm", "human_aa", "viral_aa")


@dataclass
class TrainingResult:
    history: list[dict[str, Any]] = field(default_factory=list)
    best_epoch: int | None = None
    best_metrics: dict[str, Any] = field(default_factory=dict)
    best_checkpoint: str | None = None
    stopped_early: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "history": self.history,
            "best_epoch": self.best_epoch,
            "best_metrics": self.best_metrics,
            "best_checkpoint": self.best_checkpoint,
            "stopped_early": self.stopped_early,
        }


def _modalities(batch: dict[str, Any]) -> list[torch.Tensor]:
    return [batch[key] for key in MODALITY_KEYS]


def predict_loader(
    model: HVPPIClassifier,
    loader: DataLoader,
    device: torch.device,
    *,
    with_labels: bool = True,
) -> dict[str, Any]:
    """Run inference over a loader and collect probabilities and identifiers."""
    model.eval()
    probabilities: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    pair_ids: list[str] = []
    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            outputs = model(_modalities(batch), with_discriminator=False, sample=False)
            probabilities.append(
                torch.softmax(outputs["logits"], dim=-1)[:, 1].detach().cpu().numpy()
            )
            labels.append(batch["label"].detach().cpu().numpy())
            pair_ids.extend(batch["pair_id"])
    if not probabilities:
        return {"probabilities": np.zeros(0), "labels": np.zeros(0), "pair_ids": []}
    result = {
        "probabilities": np.concatenate(probabilities),
        "labels": np.concatenate(labels),
        "pair_ids": pair_ids,
    }
    return result


def _classifier_epoch(
    model: HVPPIClassifier,
    loader: DataLoader,
    device: torch.device,
    config: dict[str, Any],
    optimizer: torch.optim.Optimizer | None,
    grad_clip: float = 0.0,
) -> dict[str, Any]:
    training = optimizer is not None
    model.train(training)
    totals = {"total": 0.0, "cross_entropy": 0.0, "contrastive": 0.0}
    probabilities: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    n_batches = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in loader:
            batch = move_batch(batch, device)
            outputs = model(
                _modalities(batch),
                sample=None if training else False,
                with_discriminator=False,
            )
            objective = classification_objective(
                outputs["logits"],
                batch["label"],
                outputs["embedding"],
                contrastive_weight=float(config["loss"]["contrastive_weight"]),
                contrastive_temperature=float(config["loss"]["contrastive_temperature"]),
            )
            if training:
                optimizer.zero_grad(set_to_none=True)
                objective["total"].backward()
                if grad_clip and grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
            for key in totals:
                totals[key] += float(objective[key].detach())
            probabilities.append(
                torch.softmax(outputs["logits"], dim=-1)[:, 1].detach().cpu().numpy()
            )
            labels.append(batch["label"].detach().cpu().numpy())
            n_batches += 1
    n_batches = max(n_batches, 1)
    result = {key: value / n_batches for key, value in totals.items()}
    result["labels"] = np.concatenate(labels) if labels else np.zeros(0)
    result["probabilities"] = np.concatenate(probabilities) if probabilities else np.zeros(0)
    return result


def _pretrain_epoch(
    model: HVPPIClassifier,
    loader: DataLoader,
    device: torch.device,
    config: dict[str, Any],
    optimizer: torch.optim.Optimizer | None,
    grad_clip: float = 0.0,
) -> dict[str, Any]:
    training = optimizer is not None
    model.train(training)
    totals: dict[str, float] = {}
    n_batches = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in loader:
            batch = move_batch(batch, device)
            modalities = _modalities(batch)
            outputs = model(modalities, with_discriminator=True, sample=None if training else False)
            terms = pretrain_loss(
                outputs,
                modalities,
                config=config,
                modality_weights=config["loss"].get("modality_weights"),
            )
            if training:
                optimizer.zero_grad(set_to_none=True)
                terms["total"].backward()
                if grad_clip and grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
            for key, value in terms.items():
                totals[key] = totals.get(key, 0.0) + float(value.detach())
            n_batches += 1
    n_batches = max(n_batches, 1)
    return {key: value / n_batches for key, value in totals.items()}


def _validate_initialization(payload, config, feature_spec):
    from ..checkpoints import CheckpointError
    from ..inference.predict import check_aaindex_matches, check_feature_backend

    for key in ("training_fold", "split_fingerprint"):
        expected = config.get("run", {}).get(key)
        actual = payload.get(key)
        if expected is not None and actual != expected:
            raise CheckpointError(f"Initialization {key} mismatch: {actual!r} != {expected!r}")
    data = payload.get("data_fingerprint")
    if data and data != feature_spec.get("records_fingerprint"):
        raise CheckpointError("Initialization data fingerprint mismatch")
    check_feature_backend(payload, feature_spec.get("sequence_backend", {}))
    check_aaindex_matches(payload, feature_spec.get("aaindex", {}))


def _loader_rng(loader):
    generator = getattr(loader, "generator", None)
    return generator.get_state() if generator is not None else None


def _save_training_progress(run_dir, optimizer, epoch, result, best_score, plateau_epochs,
                            train_loader, val_loader, *, stage, monitor):
    from ..resources.bundle import sha256

    selection = result.to_dict()
    best_path = Path(result.best_checkpoint) if result.best_checkpoint else None
    # Relative to last.pt, so the run can be moved or resumed into a new directory.
    selection["best_checkpoint"] = best_path.name if best_path else None
    save_train_state(
        run_dir.checkpoint("last_train_state.pt"), optimizer, epoch,
        resume_version=1, stage=stage, monitor=monitor,
        model_sha256=sha256(run_dir.checkpoint("last.pt")),
        best_sha256=sha256(best_path) if best_path else None,
        best_score=best_score, plateau_epochs=plateau_epochs, result=selection,
        loader_rng={"train": _loader_rng(train_loader), "val": _loader_rng(val_loader)},
    )


def _resume_training(path, model, optimizer, config, feature_spec, run_dir,
                     train_loader, val_loader, *, stage, monitor):
    from ..resources.bundle import sha256

    path = Path(path)
    payload = load_checkpoint(path)
    _validate_initialization(payload, config, feature_spec)
    state_path = path.with_name("last_train_state.pt")
    if not state_path.is_file():
        raise CheckpointError("Exact resume requires the matching last_train_state.pt next to last.pt")
    state = torch.load(state_path, map_location="cpu", weights_only=True)
    if state.get("resume_version") != 1:
        raise CheckpointError("Training state lacks complete resume metadata; start a new training run")
    if state.get("epoch") != payload.get("epoch") or state.get("model_sha256") != sha256(path):
        raise CheckpointError("Resume model and optimizer state do not match; use the paired last.pt")
    if state.get("stage") != stage or payload.get("stage") != stage or state.get("monitor") != monitor:
        raise CheckpointError("Resume training stage or selection monitor mismatch")
    result = TrainingResult(**state["result"])
    if result.best_checkpoint:
        name = result.best_checkpoint
        if Path(name).name != name:
            raise CheckpointError("Invalid best checkpoint name in training state")
        source = path.with_name(name)
        if not source.is_file() or sha256(source) != state.get("best_sha256"):
            raise CheckpointError("Resume requires the original selected best checkpoint")
        if run_dir is not None:
            destination = run_dir.checkpoint(name)
            if source.resolve() != destination.resolve():
                shutil.copyfile(source, destination)
            result.best_checkpoint = str(destination)
        else:
            result.best_checkpoint = str(source)
    model.load_state_dict(payload["state_dict"])
    # Restore RNGs after model initialization and before the next loader iteration.
    load_train_state(state_path, optimizer)
    for name, loader in (("train", train_loader), ("val", val_loader)):
        rng = state["loader_rng"][name]
        generator = getattr(loader, "generator", None)
        if (rng is None) != (generator is None):
            raise CheckpointError("Resume DataLoader generator configuration mismatch")
        if generator is not None:
            generator.set_state(rng)
    return int(state["epoch"]), state["best_score"], state["plateau_epochs"], result


def train_pretrain(
    model: HVPPIClassifier,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    config: dict[str, Any],
    device: torch.device,
    feature_spec: dict[str, Any],
    split_fingerprint: str | None = None,
    run_dir: RunDirectory | None = None,
    epochs: int | None = None,
    resume_from: str | None = None,
) -> TrainingResult:
    logger = get_logger()
    settings = config["train_pretrain"]
    epochs = int(epochs if epochs is not None else settings["epochs"])
    seed_everything(int(config["run"]["seed"]), deterministic=bool(config["run"]["deterministic"]))
    optimizer = torch.optim.Adam(
        list(pretrain_parameters(model)),
        lr=float(settings["lr"]),
        weight_decay=float(settings["weight_decay"]),
    )
    result = TrainingResult()
    start_epoch = 0
    best_score = math.inf
    if resume_from:
        start_epoch, best_score, _, result = _resume_training(
            resume_from, model, optimizer, config, feature_spec, run_dir,
            train_loader, val_loader, stage="pretrain", monitor="val_loss",
        )
        logger.info("Resuming pretraining at epoch %d", start_epoch)

    for epoch in range(start_epoch, epochs):
        train_stats = _pretrain_epoch(
            model, train_loader, device, config, optimizer, float(settings.get("grad_clip", 0.0))
        )
        val_stats = _pretrain_epoch(model, val_loader, device, config, None)
        record = {"epoch": epoch + 1, "train": train_stats, "val": val_stats}
        result.history.append(record)
        logger.info(
            "pretrain epoch %d/%d train_loss=%.4f val_loss=%.4f kl=%.4f",
            epoch + 1,
            epochs,
            train_stats.get("total", float("nan")),
            val_stats.get("total", float("nan")),
            val_stats.get("kl", float("nan")),
        )
        if run_dir is not None:
            save_checkpoint(
                run_dir.checkpoint("last.pt"),
                model,
                model_name="deephvi_pretrain",
                config=config,
                feature_spec=feature_spec,
                split_fingerprint=split_fingerprint,
                seed=int(config["run"]["seed"]),
                epoch=epoch + 1,
                metrics={"val_loss": val_stats.get("total", float("nan"))},
                extra={"stage": "pretrain"},
            )
        score = val_stats.get("total", math.inf)
        if score < best_score:
            best_score = score
            result.best_epoch = epoch + 1
            result.best_metrics = dict(val_stats)
            if run_dir is not None:
                path = save_checkpoint(
                    run_dir.checkpoint("best_pretrain.pt"),
                    model,
                    model_name="deephvi_pretrain",
                    config=config,
                    feature_spec=feature_spec,
                    split_fingerprint=split_fingerprint,
                    seed=int(config["run"]["seed"]),
                    epoch=epoch + 1,
                    metrics={"val_loss": score},
                    extra={"stage": "pretrain"},
                )
                result.best_checkpoint = str(path)
        if run_dir is not None:
            _save_training_progress(run_dir, optimizer, epoch + 1, result, best_score, 0,
                                    train_loader, val_loader, stage="pretrain", monitor="val_loss")
    return result


def train_classifier(
    model: HVPPIClassifier,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    config: dict[str, Any],
    device: torch.device,
    feature_spec: dict[str, Any],
    split_fingerprint: str | None = None,
    run_dir: RunDirectory | None = None,
    epochs: int | None = None,
    resume_from: str | None = None,
    pretrain_checkpoint: str | None = None,
) -> TrainingResult:
    logger = get_logger()
    settings = config["train_classifier"]
    epochs = int(epochs if epochs is not None else settings["epochs"])
    seed_everything(int(config["run"]["seed"]), deterministic=bool(config["run"]["deterministic"]))
    if pretrain_checkpoint and not resume_from:
        payload = load_checkpoint(pretrain_checkpoint)
        _validate_initialization(payload, config, feature_spec)
        model.load_state_dict(payload["state_dict"])
        logger.info("Initialised encoder from %s", pretrain_checkpoint)

    optimizer = torch.optim.Adam(
        parameter_groups(
            model,
            lr=float(settings["lr"]),
            encoder_lr_scale=float(settings.get("encoder_lr_scale", 1.0)),
            weight_decay=float(settings["weight_decay"]),
        )
    )
    monitor = str(settings.get("monitor", "val_auc"))
    maximize = monitor in {
        "val_auc",
        "val_roc_auc",
        "val_accuracy",
        "val_f1",
        "val_recall",
        "val_precision",
    }
    patience = settings.get("early_stopping_patience")
    plateau_epochs = 0
    best_score = -math.inf if maximize else math.inf
    result = TrainingResult()
    start_epoch = 0
    if resume_from:
        start_epoch, best_score, plateau_epochs, result = _resume_training(
            resume_from, model, optimizer, config, feature_spec, run_dir,
            train_loader, val_loader, stage="classifier", monitor=monitor,
        )
        logger.info("Resuming classifier training at epoch %d", start_epoch)
        if result.stopped_early:
            return result

    for epoch in range(start_epoch, epochs):
        train_stats = _classifier_epoch(
            model, train_loader, device, config, optimizer, float(settings.get("grad_clip", 0.0))
        )
        val_stats = _classifier_epoch(model, val_loader, device, config, None)
        threshold = float(config["eval"]["threshold"])
        average = str(config["eval"]["average"])
        train_metrics = binary_classification_metrics(
            train_stats["labels"], train_stats["probabilities"], threshold=threshold, average=average
        )
        val_metrics = binary_classification_metrics(
            val_stats["labels"], val_stats["probabilities"], threshold=threshold, average=average
        )
        record = {
            "epoch": epoch + 1,
            "train_loss": train_stats["total"],
            "train_metrics": train_metrics,
            "val_loss": val_stats["total"],
            "val_metrics": val_metrics,
        }
        result.history.append(record)
        logger.info(
            "epoch %d/%d train_loss=%.4f val_loss=%.4f %s",
            epoch + 1,
            epochs,
            train_stats["total"],
            val_stats["total"],
            format_metrics(val_metrics),
        )
        if run_dir is not None:
            save_checkpoint(
                run_dir.checkpoint("last.pt"),
                model,
                model_name="deephvi",
                config=config,
                feature_spec=feature_spec,
                split_fingerprint=split_fingerprint,
                seed=int(config["run"]["seed"]),
                epoch=epoch + 1,
                metrics=val_metrics,
                extra={"stage": "classifier"},
            )

        score = _monitor_value(monitor, val_metrics, val_stats["total"])
        improved = score > best_score if maximize else score < best_score
        if improved and not (isinstance(score, float) and math.isnan(score)):
            best_score = score
            plateau_epochs = 0
            result.best_epoch = epoch + 1
            result.best_metrics = val_metrics
            if run_dir is not None:
                path = save_checkpoint(
                    run_dir.checkpoint("best.pt"),
                    model,
                    model_name="deephvi",
                    config=config,
                    feature_spec=feature_spec,
                    split_fingerprint=split_fingerprint,
                    seed=int(config["run"]["seed"]),
                    epoch=epoch + 1,
                    metrics=val_metrics,
                    extra={"stage": "classifier"},
                )
                result.best_checkpoint = str(path)
        else:
            plateau_epochs += 1
            if patience is not None and plateau_epochs >= int(patience):
                logger.info("Early stopping after %d epochs without improvement", plateau_epochs)
                result.stopped_early = True
        if run_dir is not None:
            _save_training_progress(run_dir, optimizer, epoch + 1, result, best_score, plateau_epochs,
                                    train_loader, val_loader, stage="classifier", monitor=monitor)
        if result.stopped_early:
            break
    return result


def _monitor_value(monitor: str, metrics: dict[str, Any], val_loss: float) -> float:
    if monitor == "val_loss":
        return float(val_loss)
    if monitor.startswith("val_"):
        key = monitor[4:]
        if key == "auc":
            key = "roc_auc"
        value = metrics.get(key)
        return float(value) if value is not None else float("nan")
    raise ValueError(f"Unknown monitor: {monitor!r}")
