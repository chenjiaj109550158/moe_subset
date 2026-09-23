"""Freeze seeded IDs without selecting by baseline correctness."""

from __future__ import annotations

import json
import random
import re
import urllib.request
from pathlib import Path
from typing import Any

from pseudoroute.restart.state import digest, write_json


def prepare_data(out: Path, cache: Path) -> dict[str, Any]:
    from datasets import load_dataset  # type: ignore[import-untyped]

    meta_url = "https://huggingface.co/api/datasets/openai/gsm8k"
    with urllib.request.urlopen(meta_url, timeout=30) as response:
        meta = json.load(response)
    revision = meta["sha"]
    dataset = load_dataset("openai/gsm8k", "main", revision=revision, cache_dir=str(cache))
    if len(dataset["train"]) != 7473 or len(dataset["test"]) != 1319:
        raise ValueError("unexpected GSM8K split size")
    excluded = set()
    sources = []
    for path in sorted(Path("configs/benchmark").glob("*samples*.json")):
        if not any(s in path.name for s in ("pseudo", "qwen", "subset")):
            continue
        ids = {int(x) for x in re.findall(r'"test-(\d+)"', path.read_text())}
        excluded.update(ids)
        if ids:
            sources.append({"path": str(path), "sha256": digest(path), "ids": sorted(ids)})
    rng = random.Random(20260922)
    train = list(range(7473))
    rng.shuffle(train)
    confirm = [i for i in range(1319) if i not in excluded]
    rng.shuffle(confirm)
    if len(confirm) < 264:
        raise ValueError("insufficient disjoint confirmation pool")
    manifest = {
        "dataset_id": "openai/gsm8k",
        "revision": revision,
        "seed": 20260922,
        "development_ids": [f"train-{i}" for i in train[:32]],
        "confirmation_ids": [f"test-{i}" for i in confirm[:256]],
        "confirmation_performance_ids": [f"test-{i}" for i in confirm[256:264]],
        "excluded_policy_development_ids": sorted(excluded),
        "exclusion_sources": sources,
        "few_shot_demonstrations": [],
        "scope": "confirmation split; historical baseline evaluation disclosed",
        "selection_uses_correctness": False,
        "actual_cache_file_bytes": sum(p.stat().st_size for p in cache.rglob("*") if p.is_file()),
        "maximum_dataset_download_gib": 2,
    }
    evaluator = []
    prompts = []
    for split, selected_ids in [("train", train[:32]), ("test", confirm[:264])]:
        for i in selected_ids:
            row = dataset[split][i]
            sid = f"{split}-{i}"
            prompts.append(
                {
                    "sample_id": sid,
                    "user_prompt": f"Q: {row['question']}\nA: Let's think step by step.",
                }
            )
            evaluator.append({"sample_id": sid, "target": row["answer"].split("####")[-1].strip()})
    write_json(out / "samples_manifest.json", manifest)
    write_json(out / "prompts.json", prompts)
    write_json(out / "evaluator_targets.json", evaluator)
    return manifest
