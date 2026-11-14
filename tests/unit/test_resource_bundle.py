import json

import pytest

from deephvi.resources.bundle import ResourceError, resolve_resource, verify_bundle
from deephvi.resources.packaging import collect_resources


def make_bundle(tmp_path):
    source = tmp_path / "source.txt"
    source.write_text("model")
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({"version": 1, "missing_required": ["server_training_csv"],
                                "files": [{"source": "source.txt", "path": "models/server_run/classifier.pt"}]}))
    root = tmp_path / "resources"
    collect_resources(spec, root)
    return root, spec


def test_integrity_and_release_gate(tmp_path):
    root, _ = make_bundle(tmp_path)
    assert verify_bundle(root)["verified_files"] == 1
    with pytest.raises(ResourceError, match="Not a formal release"):
        verify_bundle(root, require_release=True)
    path = resolve_resource("checkpoint", resources_dir=root)
    path.write_text("other")
    with pytest.raises(ResourceError, match="mismatch"):
        verify_bundle(root)
    with pytest.raises(ResourceError, match="mismatch"):
        resolve_resource("checkpoint", resources_dir=root)


def test_resource_precedence_and_no_overwrite(tmp_path):
    root, spec = make_bundle(tmp_path)
    explicit = tmp_path / "explicit.pt"
    explicit.write_text("explicit")
    configured = tmp_path / "configured.pt"
    configured.write_text("configured")
    assert resolve_resource("checkpoint", explicit=explicit, configured=configured, resources_dir=root) == explicit
    assert resolve_resource("checkpoint", configured=configured, resources_dir=root) == configured
    (tmp_path / "source.txt").write_text("changed")
    with pytest.raises(ResourceError, match="overwrite"):
        collect_resources(spec, root)
    assert (root / "models/server_run/classifier.pt").read_text() == "model"


def test_manifest_cannot_escape_bundle(tmp_path):
    root, spec = make_bundle(tmp_path)
    payload = json.loads(spec.read_text())
    payload["files"][0]["path"] = "../escaped"
    spec.write_text(json.dumps(payload))
    with pytest.raises(ResourceError, match="Unsafe"):
        collect_resources(spec, root)
    assert not (tmp_path / "escaped").exists()


def test_training_defaults_and_doctor_use_supplied_csv_names(tmp_path):
    from deephvi.doctor import collect_doctor_report
    from deephvi.resources.bundle import DEFAULTS

    names = ["SH_SV_feature_with_vectors.csv", "SH_SV_feature_with_vectors_0.5.csv"]
    assert DEFAULTS["pairs"] == "training/" + names[1]
    assert DEFAULTS["pretrain_pairs"] == "training/" + names[0]
    folder = tmp_path / "data/training"
    folder.mkdir(parents=True)
    for name in names:
        (folder / name).write_text("S_H,S_V\nACDE,FGHI\n")
    report = collect_doctor_report(tmp_path)
    checks = {entry["id"]: entry for entry in report["checks"]}
    assert "full_dataset" not in checks
    for name in names:
        assert checks[f"training_csv_{name}"]["status"] == "available"
    (folder / names[0]).unlink()
    checks = {entry["id"]: entry for entry in collect_doctor_report(tmp_path)["checks"]}
    assert checks[f"training_csv_{names[0]}"]["status"] == "missing"


def test_cli_detects_data_resource_root(tmp_path, monkeypatch):
    from deephvi.cli import _config, build_parser

    root = tmp_path / "data"
    source = tmp_path / "model-config.json"
    source.write_text("{}")
    spec = tmp_path / "collect.json"
    spec.write_text(json.dumps({"version": "v1", "files": [{"source": str(source), "path": "models/esm2/config.json"}]}))
    collect_resources(spec, root)
    monkeypatch.chdir(tmp_path)
    args = build_parser().parse_args(["predict"])
    config = _config(args)
    assert config["resources"]["dir"] == "data"
    assert config["data"]["esm"]["model"] == str(root / "models/esm2")
    assert verify_bundle(root, require_release=True)["status"] == "resource_bundle"


def test_archive_under_resource_root_contains_only_inventory(tmp_path):
    import tarfile

    from deephvi.resources.packaging import archive_resources

    root, _ = make_bundle(tmp_path)
    unrelated = root / "scratch.txt"
    unrelated.write_text("unlisted")
    archive = root / "DeepHVI_resources_v1.tar.gz"
    archive_resources(root, archive)
    with tarfile.open(archive) as handle:
        names = handle.getnames()
    assert names == ["resources/models/server_run/classifier.pt", "resources/manifest.json", "resources/SHA256SUMS"]
    assert archive.with_name(archive.name + ".sha256").exists()


def test_default_profile_and_backend_override_keep_external_bundle(tmp_path, monkeypatch):
    from deephvi.cli import _config, build_parser

    profile = tmp_path / "default.yaml"
    profile.write_text("data:\n  esm:\n    backend: esm2\n    model: esm2\n")
    external = tmp_path / "external"
    monkeypatch.chdir(tmp_path)
    args = build_parser().parse_args(["prepare-features", "--config", str(profile),
                                      "--resources-dir", str(external), "--backend", "esm2",
                                      "--pairs", "pairs.csv", "--out", "features"])
    resolved = _config(args)
    assert resolved["data"]["esm"]["model"] == str(external / "models/esm2")


def test_data_inspection_does_not_need_esm_files(tmp_path, monkeypatch):
    from deephvi.cli import main

    monkeypatch.chdir(tmp_path)
    root = tmp_path / "data"
    root.mkdir()
    (root / "manifest.json").write_text('{"files": []}')
    pairs = tmp_path / "pairs.csv"
    pairs.write_text("human_seq,viral_seq,label\nACDE,FGHI,1\n")
    assert main(["inspect-data", "--input", str(pairs), "--count"]) == 0
    assert main(["config", "show", "--set", "data.esm.backend=hash"]) == 0


def test_python_esm_defaults_use_unified_data_and_check_integrity(tmp_path):
    from deephvi.data.esm import ESM2Backend

    source = tmp_path / "config.json"
    source.write_text("{}")
    spec = tmp_path / "collect.json"
    spec.write_text(json.dumps({"files": [{"source": str(source), "path": "models/esm2/config.json"}]}))
    collect_resources(spec, tmp_path / "data")
    model = tmp_path / "data/models/esm2"
    assert ESM2Backend._resolve_source("esm2", tmp_path) == str(model)
    (model / "config.json").write_text("damaged")
    with pytest.raises(ResourceError, match="mismatch"):
        ESM2Backend._resolve_source(str(model), tmp_path)
