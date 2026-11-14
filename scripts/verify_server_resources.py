"""CPU-only, bounded numerical acceptance of a supplied server resource bundle."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from deephvi.cli import main as cli
from deephvi.data.aaindex import encode_many
from deephvi.data.server import ServerSequenceBackend
from deephvi.inference.predict import predict_arrays
from deephvi.io import write_json
from deephvi.models.registry import load_model_from_checkpoint
from deephvi.paths import environment_info
from deephvi.resources.bundle import verify_bundle


def verify(root, out):
    torch.set_num_threads(2)
    root, out = Path(root), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    integrity = verify_bundle(root)
    model, payload = load_model_from_checkpoint(str(root / "models/server_run/classifier.pt"))
    config = payload["config"]
    config["resources"]["dir"] = str(root)
    config["data"]["esm"].update(model=None, batch_size=2)
    backend = ServerSequenceBackend(config)
    sequences = sorted(backend.cache, key=lambda seq: (len(seq), seq))
    selected = [sequences[0], sequences[len(sequences) // 2], sequences[-1]]
    cached = backend.encode(selected)
    for seq in selected:
        del backend.cache[seq]
    recomputed = backend.encode(selected)
    np.testing.assert_allclose(recomputed, cached, atol=2e-5, rtol=2e-5)
    legacy_cache = torch.load(root / "cache/seq_emb_cache.pt", map_location="cpu", weights_only=True)["cache"]
    longest = max(legacy_cache, key=len)
    assert longest not in backend.cache
    selected_with_long = selected + [longest]
    encoded_with_long = backend.encode(selected_with_long)
    raw = backend._encoder()
    with torch.inference_mode():
        # Independent attention-mean reference using the current 1024-token cap.
        tokens = raw.tokenizer(selected_with_long, return_tensors="pt", padding=True, truncation=True, max_length=1024)
        hidden = raw.model(**tokens).last_hidden_state
        mask = tokens["attention_mask"].unsqueeze(-1)
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
        reference = torch.nn.functional.linear(pooled, backend.projection_state["weight"], backend.projection_state["bias"]).numpy()
    np.testing.assert_allclose(encoded_with_long, reference, atol=2e-5, rtol=2e-5)
    token_lengths = {length: len(raw.tokenizer("A" * length, truncation=True, max_length=raw.max_length)["input_ids"])
                     for length in (510, 511, 1022, 1023)}
    assert token_lengths == {510: 512, 511: 513, 1022: 1024, 1023: 1024}
    aa = encode_many(selected)
    p_cached = predict_arrays(model, [cached, cached[::-1].copy(), aa, aa[::-1].copy()], device=torch.device("cpu"))
    p_fresh = predict_arrays(model, [recomputed, recomputed[::-1].copy(), aa, aa[::-1].copy()], device=torch.device("cpu"))
    np.testing.assert_allclose(p_cached, p_fresh, atol=2e-5, rtol=2e-5)
    novel = "MACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWY"
    assert novel not in backend.cache
    result = backend.encode([novel])
    assert result.shape == (1, 256) and np.isfinite(result).all()
    for name in ["cached_pairs", "new_pairs"]:
        assert cli(["predict", "--resources-dir", str(root), "--pairs", str(root / f"examples/inputs/{name}.csv"),
                    "--out", str(out / f"{name}.csv"), "--device", "cpu"]) == 0
    report = {"scope": "CPU lightweight code verification, no full training or fine-tuning", "environment": environment_info(),
              "resource_integrity": integrity, "cache_sequences": len(sequences), "checked_lengths": list(map(len, selected_with_long)),
              "max_length": raw.max_length, "token_lengths": token_lengths,
              "legacy_cache_entries_skipped": len(legacy_cache) - len(sequences),
              "cache_recompute_max_abs_error": float(np.max(np.abs(recomputed - cached))),
              "reference_encoder_max_abs_error": float(np.max(np.abs(encoded_with_long - reference))),
              "cached_vs_recomputed_probability_error": float(np.max(np.abs(p_cached - p_fresh))),
              "novel_sequence_cache_miss": "passed", "sample_predictions": "passed"}
    write_json(out / "server_verification.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resources-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.resources_dir, args.out), indent=2))


if __name__ == "__main__":
    main()
