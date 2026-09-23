"""Acquire only the authorized checkpoint after cache, metadata and capacity checks."""

import argparse
import hashlib
import json
import shutil
import urllib.request
from pathlib import Path

MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"
REVISION = "0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    p.add_argument("--cache", required=True)
    p.add_argument("--download", action="store_true")
    a = p.parse_args()
    out, cache = Path(a.run_dir), Path(a.cache)
    cache.mkdir(parents=True, exist_ok=True)
    url = f"https://huggingface.co/api/models/{MODEL}/revision/{REVISION}?blobs=true"
    with urllib.request.urlopen(url, timeout=30) as response:
        meta = json.load(response)
    assert meta["sha"] == REVISION and not meta["gated"]
    files = [
        f
        for f in meta["siblings"]
        if f["rfilename"].endswith((".safetensors", ".json", ".txt"))
        and f["rfilename"] != "config_1m.json"
    ]
    size = sum(f["size"] for f in files)
    free = shutil.disk_usage(cache).free
    assert size <= 80 * 2**30
    assert free > size + (20 + 20 + 16) * 2**30
    manifest = dict(
        model_id=MODEL,
        revision=REVISION,
        gated=meta["gated"],
        metadata_url=url,
        files=files,
        download_bytes=size,
        free_disk_bytes=free,
        estimated_environment_bytes=20 * 2**30,
        artifact_cap_bytes=16 * 2**30,
        disk_headroom_bytes=20 * 2**30,
        weights_verified=False,
        state="METADATA_VERIFIED",
        cache=str(cache.resolve()),
        existing_cache_checks={
            str(x): x.exists()
            for x in [
                Path("/home/dev/.cache/huggingface/hub") / ("models--" + MODEL.replace("/", "--")),
                Path("/home/dev/moe_subset/.cache/pseudoroute/huggingface/hub")
                / ("models--" + MODEL.replace("/", "--")),
            ]
        },
    )
    (out / "model_manifest.json").write_text(json.dumps(manifest, indent=2))
    base = f"https://huggingface.co/{MODEL}/resolve/{REVISION}/"
    for name in ["config.json", "model.safetensors.index.json", "generation_config.json"]:
        with urllib.request.urlopen(base + name, timeout=60) as response:
            (out / ("checkpoint_" + name)).write_bytes(response.read())
    config = json.loads((out / "checkpoint_config.json").read_text())
    expected = {
        "num_hidden_layers": 48,
        "num_experts": 128,
        "num_experts_per_tok": 8,
        "hidden_size": 2048,
        "moe_intermediate_size": 768,
    }
    assert all(config[k] == v for k, v in expected.items()), config
    print(
        json.dumps({"state": manifest["state"], "checkpoint_gib": size / 2**30, "config": config}),
        flush=True,
    )
    if a.download:
        from huggingface_hub import snapshot_download

        location = snapshot_download(
            MODEL,
            revision=REVISION,
            cache_dir=cache,
            allow_patterns=[f["rfilename"] for f in files],
            max_workers=4,
        )
        checks = []
        for f in files:
            path = Path(location) / f["rfilename"]
            assert path.stat().st_size == f["size"]
            h = hashlib.file_digest(path.open("rb"), "sha256").hexdigest()
            if f.get("lfs"):
                assert h == f["lfs"]["sha256"], f["rfilename"]
            checks.append({"name": f["rfilename"], "sha256": h, "bytes": path.stat().st_size})
        manifest.update(
            state="VERIFIED", weights_verified=True, snapshot_path=location, verified_files=checks
        )
        (out / "model_manifest.json").write_text(json.dumps(manifest, indent=2))
        print("VERIFIED " + location, flush=True)


if __name__ == "__main__":
    main()
