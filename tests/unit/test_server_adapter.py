import numpy as np
import pytest
import torch

from deephvi.config import load_config
from deephvi.data.server import ServerSequenceBackend
from deephvi.inference.legacy_export import export_legacy_checkpoint
from deephvi.models.registry import load_model_from_checkpoint
from deephvi.models.server import ServerClassifier
from deephvi.resources.bundle import ResourceError


def resources(tmp_path):
    projection = tmp_path / "projection.pt"
    state = torch.nn.Linear(320, 256).state_dict()
    torch.save({"proj_state": state}, projection)
    cache = tmp_path / "cache.pt"
    torch.save({"proj_state": state, "cache": {"ACD": torch.ones(256)}}, cache)
    config = load_config()
    config["resources"] = {"paths": {"projection": str(projection), "cache": str(cache)}}
    return config, state, projection, cache


def test_cache_miss_uses_saved_projection(tmp_path, monkeypatch):
    config, state, _, _ = resources(tmp_path)
    backend = ServerSequenceBackend(config)

    class Encoder:
        def encode(self, sequences, **kwargs):
            assert sequences == ["EFG"]
            return np.ones((1, 320), dtype=np.float32)

    monkeypatch.setattr(backend, "_encoder", lambda: Encoder())
    actual = backend.encode(["ACD", "EFG", "EFG"])
    expected = torch.nn.functional.linear(torch.ones(320), state["weight"], state["bias"]).numpy()
    np.testing.assert_array_equal(actual[0], np.ones(256))
    np.testing.assert_allclose(actual[1], expected, atol=1e-6)
    np.testing.assert_array_equal(actual[1], actual[2])


def test_mismatched_cache_projection_is_rejected(tmp_path):
    config, state, _, cache = resources(tmp_path)
    state["bias"] += 1
    torch.save({"proj_state": state, "cache": {}}, cache)
    with pytest.raises(ResourceError, match="disagrees"):
        ServerSequenceBackend(config)


@pytest.mark.parametrize("cache_length", [None, 512, 1024])
def test_truncated_legacy_cache_is_recomputed_at_1024(tmp_path, monkeypatch, cache_length):
    config, state, _, cache = resources(tmp_path)
    sequences = ["A" * 510, "C" * 511, "D" * 1022, "E" * 1023]
    blob = {"proj_state": state, "cache": {seq: torch.ones(256) for seq in sequences}}
    if cache_length is not None:
        blob["encoding"] = {"max_length": cache_length}
    torch.save(blob, cache)
    esm_dir = tmp_path / "esm"
    esm_dir.mkdir()
    config["resources"]["paths"]["esm"] = str(esm_dir)
    # A historical config must not reinstate the old truncation length.
    config["data"]["esm"]["max_length"] = 512
    calls = []

    class Encoder:
        def __init__(self, **kwargs):
            assert kwargs["max_length"] == 1024

        def encode(self, missing, **kwargs):
            calls.extend(missing)
            return np.zeros((len(missing), 320), dtype=np.float32)

        def spec(self):
            return {"model_fingerprints": {}}

    monkeypatch.setattr("deephvi.data.server.ESM2Backend", Encoder)
    backend = ServerSequenceBackend(config)
    actual = backend.encode(sequences)
    if cache_length == 1024:
        assert calls == []
        np.testing.assert_array_equal(actual, np.ones((4, 256)))
    else:
        assert calls == sequences[1:]
        np.testing.assert_array_equal(actual[0], np.ones(256))
        np.testing.assert_allclose(actual[1:], np.tile(state["bias"].numpy(), (3, 1)))
    assert backend.spec()["max_length"] == 1024


def test_safe_server_conversion_roundtrip(tmp_path):
    _, _, projection, cache = resources(tmp_path)
    original = ServerClassifier().eval()
    source = tmp_path / "source.pt"
    torch.save(original.state_dict(), source)
    report = export_legacy_checkpoint(source, tmp_path / "export", projection=projection, cache=cache, allow_unsafe_pickle=True)
    loaded, payload = load_model_from_checkpoint(report["output"])
    loaded.eval()
    assert payload["encoding"] == {"max_length": 1024, "pooling": "attention_mean", "special_tokens": True}
    inputs = [torch.randn(2, dim) for dim in original.modal_dims]
    with torch.inference_mode():
        torch.testing.assert_close(original.predict_proba(inputs), loaded.predict_proba(inputs))
    assert not list(tmp_path.glob("*.tmp"))


def test_attention_pooling_includes_special_tokens():
    from deephvi.data.esm import ESM2Backend

    backend = object.__new__(ESM2Backend)
    backend.pooling = "attention_mean"
    result = backend._pool(torch.tensor([[[1.], [3.], [5.], [100.]]]), None, torch.tensor([[1, 1, 1, 0]]))
    assert result.item() == 3


def test_export_cli_reports_server_conversion(tmp_path):
    from deephvi.cli import main

    _, _, projection, cache = resources(tmp_path)
    source = tmp_path / "source.pt"
    torch.save(ServerClassifier().state_dict(), source)
    assert main(["export-legacy", "--checkpoint", str(source), "--projection", str(projection),
                 "--cache", str(cache), "--out", str(tmp_path / "converted"), "--allow-unsafe-pickle"]) == 0


def test_prediction_config_only_overrides_explicit_values(tmp_path):
    from deephvi.cli import _config_from_payload, build_parser

    profile = tmp_path / "profile.yaml"
    profile.write_text("predict:\n  batch_size: 2\n")
    config = load_config()
    config["data"]["esm"]["max_length"] = 512
    config["data"]["esm"]["pooling"] = "attention_mean"
    args = build_parser().parse_args(["predict", "--config", str(profile)])
    resolved = _config_from_payload({"config": config}, args)
    assert resolved["data"]["esm"]["max_length"] == 512
    assert resolved["data"]["esm"]["pooling"] == "attention_mean"
    assert resolved["predict"]["batch_size"] == 2


def test_server_prediction_config_uses_1024_despite_old_overrides():
    from deephvi.cli import _config_from_payload, build_parser

    config = load_config()
    config["data"]["esm"]["max_length"] = 512
    args = build_parser().parse_args(["predict", "--set", "data.esm.max_length=512"])
    payload = {"model_name": "server_tmonet_v1", "config": config, "resource_fingerprints": {}}
    resolved = _config_from_payload(payload, args)
    assert resolved["data"]["esm"]["max_length"] == 1024


def test_legacy_pickle_conversion_does_not_require_private_sources(tmp_path, monkeypatch):
    import sys
    import types

    from deephvi.compat.tmonet import encoder
    from deephvi.inference.legacy_export import load_legacy_object

    module_name = "TMONet.model.TMO_Net_model"
    module = types.ModuleType(module_name)
    module.encoder = encoder
    # Serialize a module with the original import path, then remove that path.
    with monkeypatch.context() as context:
        for name in ("TMONet", "TMONet.model"):
            package = types.ModuleType(name)
            package.__path__ = []
            context.setitem(sys.modules, name, package)
        context.setitem(sys.modules, module_name, module)
        context.setattr(encoder, "__module__", module_name)
        original = encoder()
        original.encoder = torch.nn.Identity()
        original.mu_predictor = torch.nn.Identity()
        original.log_var_predictor = torch.nn.Identity()
        torch.save(original, tmp_path / "legacy.pt")
    monkeypatch.chdir(tmp_path)
    restored = load_legacy_object("legacy.pt")
    assert isinstance(restored, encoder)
    assert module_name not in sys.modules
    with torch.inference_mode():
        assert torch.equal(restored(torch.ones(2, 4))[1], torch.ones(2, 4))
