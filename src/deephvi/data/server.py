"""Server ESM encoding: 1024 tokens, attention mean including BOS/EOS, saved projection."""
from __future__ import annotations

import numpy as np
import torch

from ..resources.bundle import ResourceError, resolve_resource, sha256
from .esm import ESM2Backend

SERVER_MAX_LENGTH = 1024


def load_projection(path):
    blob = torch.load(path, map_location="cpu", weights_only=True)
    state = blob.get("proj_state", blob)
    if set(state) != {"weight", "bias"} or state["weight"].shape != (256, 320) or state["bias"].shape != (256,):
        raise ResourceError(f"Invalid 320→256 projection: {path}")
    if not all(torch.isfinite(v).all() for v in state.values()):
        raise ResourceError(f"Non-finite projection: {path}")
    return state


class ServerSequenceBackend:
    name = "server_esm2"
    dim = 256

    def __init__(self, config, *, device="cpu"):
        settings = config["data"]["esm"]
        resources = config.get("resources", {})
        root = resources.get("dir")
        paths = resources.get("paths", {})
        self.projection_path = resolve_resource("projection", configured=paths.get("projection"), resources_dir=root)
        self.projection_state = load_projection(self.projection_path)
        self.fingerprints = resources.get("fingerprints", {})
        expected = self.fingerprints.get("projection")
        if expected and sha256(self.projection_path) != expected:
            raise ResourceError(f"Projection fingerprint mismatch: {self.projection_path}")
        self.projection = torch.nn.Linear(320, 256)
        self.projection.load_state_dict(self.projection_state)
        self.projection.eval()
        self.cache = {}
        cache_path = resolve_resource("cache", configured=paths.get("cache"), resources_dir=root, optional=True)
        self.cache_sha256 = sha256(cache_path) if cache_path else None
        if cache_path:
            expected = self.fingerprints.get("cache")
            if expected and sha256(cache_path) != expected:
                raise ResourceError(f"Cache fingerprint mismatch: {cache_path}")
            blob = torch.load(cache_path, map_location="cpu", weights_only=True)
            if set(blob.get("proj_state", {})) != set(self.projection_state) or not all(
                torch.equal(value, blob["proj_state"][key]) for key, value in self.projection_state.items()
            ):
                raise ResourceError(f"Cache projection disagrees with {self.projection_path}: {cache_path}")
            self.cache = blob["cache"]
            if any(not isinstance(key, str) or value.shape != (256,) or not torch.isfinite(value).all() for key, value in self.cache.items()):
                raise ResourceError(f"Invalid sequence cache: {cache_path}")
            # Unversioned historical caches were generated at 512 tokens. Only
            # reuse entries whose complete sequence (plus BOS/EOS) fit there;
            # truncated entries must be recomputed at the current input length.
            cache_max_length = int(blob.get("encoding", {}).get("max_length", 512))
            if cache_max_length != SERVER_MAX_LENGTH:
                limit = min(cache_max_length, SERVER_MAX_LENGTH)
                self.cache = {seq: value for seq, value in self.cache.items() if len(seq) + 2 <= limit}
        self.settings = settings
        self.root = root
        self.paths = paths
        self.device = device
        self.encoder = None

    def _encoder(self):
        if self.encoder is None:
            model = resolve_resource("esm", configured=self.paths.get("esm") or self.settings.get("model"), resources_dir=self.root)
            for name, expected in self.fingerprints.get("esm", {}).items():
                if not (model / name).is_file() or sha256(model / name) != expected:
                    raise ResourceError(f"ESM fingerprint mismatch: {model / name}")
            self.encoder = ESM2Backend(model=str(model), pooling="attention_mean", max_length=SERVER_MAX_LENGTH,
                                       batch_size=self.settings.get("batch_size", 8), device=self.device)
        return self.encoder

    def encode(self, sequences, *, batch_size=None):
        missing = list(dict.fromkeys(s for s in sequences if s not in self.cache))
        if missing:
            raw = self._encoder().encode(missing, batch_size=batch_size)
            with torch.inference_mode():
                values = self.projection(torch.from_numpy(raw)).cpu()
            self.cache.update(zip(missing, values, strict=True))
        return np.stack([self.cache[s].numpy() for s in sequences]) if sequences else np.zeros((0, 256), dtype=np.float32)

    def spec(self):
        return {"backend": self.name, "dim": 256, "max_length": SERVER_MAX_LENGTH, "pooling": "attention_mean",
                "projection_sha256": sha256(self.projection_path), "resource_fingerprints": self.fingerprints,
                "cache_sha256": self.cache_sha256,
                "model_fingerprints": self._encoder().spec()["model_fingerprints"]}
