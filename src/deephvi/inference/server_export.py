"""Explicit conversion of a trusted server checkpoint to tensors and primitives."""
from pathlib import Path

import torch

from ..config import load_config
from ..data.server import SERVER_MAX_LENGTH, load_projection
from ..io import write_json
from ..models.server import ServerClassifier, ServerEncoder
from ..resources.bundle import ResourceError, sha256


def export_server(checkpoint, output_dir, *, projection, cache=None, esm_dir=None, project_root=None):
    from .legacy_export import extract_state_dict, load_legacy_object

    checkpoint, output = Path(checkpoint), Path(output_dir)
    # Plain tensor dictionaries do not need the pickle compatibility environment.
    try:
        obj = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except Exception:
        obj = load_legacy_object(checkpoint, project_root=project_root)
    state, object_type = extract_state_dict(obj)
    classifier = any(key.startswith("downstream_predictor.") for key in state)
    model = ServerClassifier() if classifier else ServerEncoder()
    model.load_state_dict(state, strict=True)
    model.eval()
    proj = load_projection(projection)
    fingerprints = {"source_checkpoint": sha256(checkpoint), "projection": sha256(projection)}
    report = {"object_type": object_type, "classifier": classifier, "projection_cache_equal": None}
    if cache:
        blob = torch.load(cache, map_location="cpu", weights_only=True)
        equal = set(blob["proj_state"]) == set(proj) and all(torch.equal(proj[k], blob["proj_state"][k]) for k in proj)
        if not equal:
            raise ResourceError("Cache projection and independent projection differ; conversion aborted")
        fingerprints["cache"] = sha256(cache)
        report.update(projection_cache_equal=True, cache_sequences=len(blob["cache"]))
    if esm_dir:
        esm = Path(esm_dir)
        names = ["config.json", "vocab.txt", "tokenizer_config.json", "special_tokens_map.json"]
        names.append("model.safetensors" if (esm / "model.safetensors").exists() else "pytorch_model.bin")
        fingerprints["esm"] = {name: sha256(esm / name) for name in names}
    if classifier and isinstance(obj, torch.nn.Module):
        # Original graph's forward returns the distributions used by get_embedding.
        # Avoid its get_embedding: it writes intermediate tensors into the CWD.
        obj.eval()
        generator = torch.Generator().manual_seed(42)
        inputs = [torch.randn(3, dim, generator=generator) for dim in [256, 256, 1200, 1200]]
        with torch.inference_mode():
            values, _ = obj.cross_encoders(inputs, dict(enumerate(range(4))))
            embedding = []
            for i in range(4):
                weights = [1. / torch.exp(values[j][i][2]).square() for j in range(4) if j != i]
                means = [values[j][i][1] for j in range(4) if j != i]
                poe = (1. / (1. + sum(weights))) * sum(w * mu for w, mu in zip(weights, means, strict=True))
                embedding.append((values[i][i][1] + poe) / 2)
            reference = obj.downstream_predictor(torch.cat(embedding, dim=1))
            actual = model(inputs)["logits"]
            torch.testing.assert_close(actual, reference, atol=2e-5, rtol=2e-5)
        report["reference_max_abs_logit_error"] = float((actual - reference).abs().max())
    config = load_config()
    config["data"]["esm"].update(backend="server_esm2", model=None, pooling="attention_mean", max_length=SERVER_MAX_LENGTH)
    config["model"].update(name="server_tmonet_v1", esm_dim=256, project_esm=False)
    config["resources"] = {"fingerprints": fingerprints}
    payload = {
        "format_version": 1, "model_name": "server_tmonet_v1" if classifier else "server_tmonet_pretrain_v1",
        "state_dict": state, "config": config, "encoding": {"max_length": SERVER_MAX_LENGTH, "pooling": "attention_mean", "special_tokens": True},
        "feature_spec": {"dims": dict(zip(["human_esm", "viral_esm", "human_aa", "viral_aa"], [256, 256, 1200, 1200], strict=True)),
                         "sequence_backend": {"backend": "server_esm2", "dim": 256, "max_length": SERVER_MAX_LENGTH, "pooling": "attention_mean"}},
        "resource_fingerprints": fingerprints, "training_fold": 3, "protocol": "server_historical",
        "data_fingerprint": None, "split_fingerprint": None,
        "provenance_warning": "Training CSVs are supplied separately. Exact checkpoint dataset/split fingerprints were not recorded. Historical inference used the train loader; metrics are not independent test results.",
    }
    output.mkdir(parents=True, exist_ok=True)
    destination = output / ("classifier.pt" if classifier else "pretrain.pt")
    if destination.exists():
        raise ResourceError(f"Refusing to overwrite existing conversion: {destination}")
    torch.save(payload, destination)
    report.update(output=str(destination), sha256=sha256(destination), resource_fingerprints=fingerprints,
                  exact_graph_verified="reference_max_abs_logit_error" in report,
                  deephvi_checkpoint_compatible=classifier)
    write_json(output / ("classifier_conversion.json" if classifier else "pretrain_conversion.json"), report)
    return report
