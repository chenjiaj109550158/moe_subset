import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer, StopStringCriteria

from pseudoroute.restart.stopping import RequestStopState

p = argparse.ArgumentParser()
p.add_argument("--run-dir", type=Path, required=True)
a = p.parse_args()
snapshot = (
    ".cache/restart/hub/models--Qwen--Qwen3-30B-A3B-Instruct-2507/snapshots/"
    "0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe"
)
tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
prompt = tokenizer("Q: Compute the answer.\nA:", return_tensors="pt")["input_ids"]
tokens = tokenizer.encode(" one two three four" * 40, add_special_tokens=False)[:128]
scores = torch.zeros(1, len(tokenizer))
rows = []
StopStringCriteria(tokenizer, ["Q:"])
for repetition in range(5):
    for mode in ("legacy", "request_state") if repetition % 2 == 0 else ("request_state", "legacy"):
        start = time.perf_counter()
        results = []
        if mode == "request_state":
            state = RequestStopState(tokenizer, prompt, None, ("Q:",), len(tokens))
            results = [state.append(token, scores) for token in tokens]
        else:
            for i in range(len(tokens)):
                criterion = StopStringCriteria(tokenizer, ["Q:"])
                sequence = torch.cat((prompt, torch.tensor([tokens[: i + 1]])), 1)
                results.append(bool(criterion(sequence, scores).item()))
        elapsed = time.perf_counter() - start
        rows.append(
            {
                "mode": mode,
                "repetition": repetition,
                "tokens": len(tokens),
                "wall_seconds": elapsed,
                "stop_results": results,
                "device": "cpu",
                "model_inference_included": False,
            }
        )
assert all(row["stop_results"] == rows[0]["stop_results"] for row in rows)
(a.run_dir / "profiles/stop_cpu.json").write_text(json.dumps(rows, indent=2))
