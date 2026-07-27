import os
from pathlib import Path

import pytest
import torch

from pseudoroute.analysis.dapq_factorial import capture_factorial, compare_capture
from pseudoroute.analysis.pseudo_sequences import OfflineDocument, build_factorial_sequences
from pseudoroute.execution import NaturalRoutingPolicy
from pseudoroute.models.adapters.hf_mixtral import HFMixtralAdapter
from pseudoroute.models.base import TraceLevel, TraceRequest
from pseudoroute.tracing import TraceStore, collect_sample, validate_trace

MODEL_ID = "hf-internal-testing/tiny-random-MixtralForCausalLM"
MODEL_REVISION = "ccb12fe2fc142cb752085506c3db22572290e90c"
DATASET_ID = "Salesforce/wikitext"
DATASET_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
CACHE = Path(".cache/pseudoroute/huggingface/hub")

pytestmark = pytest.mark.skipif(
    os.environ.get("PSEUDOROUTE_RUN_EXTERNAL") != "1",
    reason="set PSEUDOROUTE_RUN_EXTERNAL=1 for pinned external-model tests",
)


def load_adapter() -> HFMixtralAdapter:
    return HFMixtralAdapter.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        device="cpu",
        cache_dir=str(CACHE),
        local_files_only=True,
    )


def test_hf_native_trace_and_policy_parity(tmp_path: Path) -> None:
    from transformers import AutoTokenizer

    subject = load_adapter()
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        cache_dir=str(CACHE),
        local_files_only=True,
        trust_remote_code=False,
    )
    tokens = tokenizer("Hello MoE", return_tensors="pt")["input_ids"]
    base = subject.run_base_forward(tokens)
    traced = subject.run_base_forward(tokens, trace_request=TraceRequest(TraceLevel.ROUTER_LOGITS))
    policy = subject.forward_with_policy(tokens, NaturalRoutingPolicy())
    assert torch.equal(base.logits, traced.logits)
    assert torch.equal(base.logits, policy.logits)
    assert len(traced.traces) == tokens.numel() * subject.spec.num_layers
    with torch.inference_mode():
        native = subject.model(
            input_ids=tokens, output_router_logits=True, use_cache=False, return_dict=True
        )
    native_logits = native.router_logits[0].reshape(1, tokens.shape[1], -1).float()
    native_scores = native_logits.softmax(dim=-1)
    native_topk_scores, native_topk_ids = native_scores.topk(2, dim=-1)
    native_weights = native_topk_scores / native_topk_scores.sum(dim=-1, keepdim=True)
    first = traced.traces[0]
    assert torch.equal(first.topk_ids, native_topk_ids[:, 0].cpu())
    assert torch.equal(first.topk_weights, native_weights[:, 0].cpu())
    assert torch.equal(first.raw_logits, native_logits[:, 0].to(first.raw_logits.dtype).cpu())
    assert subject.spec.architecture == "MixtralForCausalLM"
    assert all(len(layer.experts) == 4 for layer in subject.iter_moe_layers())

    store = TraceStore.create(
        tmp_path / "real-trace",
        spec=subject.spec,
        model_revision=MODEL_REVISION,
        dataset_id=DATASET_ID,
        dataset_revision=DATASET_REVISION,
        dataset_split="validation",
        trace_level=TraceLevel.ROUTER_LOGITS,
    )
    collect_sample(
        subject,
        store,
        sample_id="wikitext-validation-0",
        token_ids=tokens,
        trace_level=TraceLevel.ROUTER_LOGITS,
    )
    store.mark_complete()
    manifest = validate_trace(tmp_path / "real-trace")
    assert manifest.model_revision == MODEL_REVISION
    assert manifest.dataset_revision == DATASET_REVISION


def test_hf_offline_factorial_capture_support() -> None:
    subject = load_adapter()
    sequence = build_factorial_sequences(
        (OfflineDocument("real", torch.tensor([1, 327, 11537, 283, 381])),),
        boundary=3,
        horizon=2,
        vocab_size=32000,
        content_seed=2,
        position_offset=3,
        include_context_swap=False,
    )[0]
    captured = capture_factorial(subject, sequence)
    assert captured.pre_rope_query is not None
    assert captured.post_rope_query is None
    assert captured.post_attention_state is not None
    assert captured.router_input is not None
    rows = compare_capture(captured, captured, horizons=(1, 2), top_k=2)
    assert all(row.value == 0.0 for row in rows if row.metric == "logit_mse")
    assert all(row.value == 1.0 for row in rows if row.metric == "topk_recall")
