from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from transformers import AutoTokenizer, StopStringCriteria

from pseudoroute.restart.stopping import RequestStopState


@pytest.fixture(scope="module")
def tokenizer():
    snapshot = Path(
        ".cache/restart/hub/models--Qwen--Qwen3-30B-A3B-Instruct-2507/snapshots/0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe"
    )
    return AutoTokenizer.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False)


@pytest.mark.parametrize(
    "prompt,text,stops",
    [
        ("A:", "hello Q: next", ("Q:",)),
        ("over", "lappingstop tail", ("overlap", "lappingstop", "stop")),
        ("x", "aaaaab tail", ("aab", "ab")),
        ("Q: prompt", "answer\nQ: next", ("Q:", "</s>")),
        ("short", "done", ()),
    ],
)
def test_native_stop_positions_and_once_per_request(tokenizer, prompt, text, stops):
    prefix = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")["input_ids"]
    tokens = tokenizer.encode(text, add_special_tokens=False)
    scores = torch.zeros(1, len(tokenizer))
    with patch("transformers.StopStringCriteria", wraps=StopStringCriteria) as constructor:
        state = RequestStopState(tokenizer, prefix, None, stops, len(tokens))
        actual = []
        for t in tokens:
            actual.append(state.append(t, scores))
        assert constructor.call_count == int(bool(stops))
    expected = []
    for i in range(len(tokens)):
        if stops:
            sequence = torch.cat((prefix, torch.tensor([tokens[: i + 1]])), 1)
            expected.append(
                bool(StopStringCriteria(tokenizer, list(stops))(sequence, scores).item())
            )
        else:
            expected.append(False)
    assert actual == expected


def test_eos_list_and_capacity(tokenizer):
    state = RequestStopState(tokenizer, torch.tensor([[1, 2]]), [5, 6], (), 2)
    assert not state.append(4, torch.zeros(1))
    assert state.append(6, torch.zeros(1))
    with pytest.raises(ValueError):
        state.append(7, torch.zeros(1))
