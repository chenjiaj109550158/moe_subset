"""Pinned task loading and prompt rendering for the accuracy suite."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pseudoroute.benchmark.config import AccuracyDatasetConfig


@dataclass(frozen=True)
class BenchmarkExample:
    task: str
    sample_id: str
    user_prompt: str
    assistant_prefix: str | None
    target: str
    row: dict[str, Any]
    stop_strings: tuple[str, ...]
    max_new_tokens: int


def _human_eval(row: dict[str, Any], config: AccuracyDatasetConfig) -> BenchmarkExample:
    prompt = str(row["prompt"])
    return BenchmarkExample(
        task=config.key,
        sample_id=str(row["task_id"]),
        user_prompt=(
            "Write a solution to the following problem and make sure that it passes the tests:\n"
            f"```python\n{prompt}\n```\n"
        ),
        assistant_prefix=f"Here is the completed function:\n```python\n{prompt}\n",
        target=f"{row['test']}\ncheck({row['entry_point']})",
        row=row,
        stop_strings=("\nclass", "\ndef", "\n#", "\nif", "\nprint", "```"),
        max_new_tokens=config.max_new_tokens,
    )


def _mbpp_plus(row: dict[str, Any], config: AccuracyDatasetConfig) -> BenchmarkExample:
    tests = list(row["test_list"])
    return BenchmarkExample(
        task=config.key,
        sample_id=str(row["task_id"]),
        user_prompt=(
            f"{row['prompt']} Your code should satisfy the following assertion:\n{tests[0]}"
        ),
        assistant_prefix="Here is a solution to this programming problem:\n```python\n",
        target=str(row["test"]),
        row=row,
        stop_strings=("```",),
        max_new_tokens=config.max_new_tokens,
    )


def _gsm8k(row: dict[str, Any], config: AccuracyDatasetConfig, index: int) -> BenchmarkExample:
    target = str(row["answer"]).split("####")[-1].strip()
    return BenchmarkExample(
        task=config.key,
        sample_id=f"test-{index}",
        user_prompt=f"Q: {row['question']}\nA: Let's think step by step.",
        assistant_prefix=None,
        target=target,
        row=row,
        stop_strings=("Q:", "</s>", "<|im_end|>"),
        max_new_tokens=config.max_new_tokens,
    )


def _aime(row: dict[str, Any], config: AccuracyDatasetConfig, index: int) -> BenchmarkExample:
    problem_key = next(key for key in row if key.lower() == "problem")
    answer_key = next(key for key in row if key.lower() == "answer")
    return BenchmarkExample(
        task=config.key,
        sample_id=f"{config.split}-{index}",
        user_prompt=f"Question: {row[problem_key]}\nAnswer:",
        assistant_prefix=None,
        target=str(row[answer_key]),
        row=row,
        stop_strings=("Question:", "</s>", "<|im_end|>", "<|eot_id|>"),
        max_new_tokens=config.max_new_tokens,
    )


def _strategyqa(row: dict[str, Any], config: AccuracyDatasetConfig, index: int) -> BenchmarkExample:
    target = "yes" if bool(row["answer"]) else "no"
    return BenchmarkExample(
        task=config.key,
        sample_id=str(row.get("qid", f"test-{index}")),
        user_prompt=(
            "Answer the following question with only yes or no.\n"
            f"Question: {row['question']}\nAnswer:"
        ),
        assistant_prefix=None,
        target=target,
        row=row,
        stop_strings=("\n", "</s>", "<|im_end|>", "<|eot_id|>"),
        max_new_tokens=config.max_new_tokens,
    )


def load_examples(config: AccuracyDatasetConfig, *, cache_dir: str) -> tuple[BenchmarkExample, ...]:
    from datasets import load_dataset  # type: ignore[import-untyped]

    dataset = load_dataset(
        config.dataset_id,
        config.config,
        split=config.split,
        revision=config.revision,
        cache_dir=cache_dir,
    )
    if len(dataset) != config.expected_samples:
        raise ValueError(
            f"{config.key}: expected {config.expected_samples} rows, got {len(dataset)}"
        )
    examples: list[BenchmarkExample] = []
    for index, raw in enumerate(dataset):
        row = dict(raw)
        if config.key == "humaneval":
            example = _human_eval(row, config)
        elif config.key == "mbpp_plus":
            example = _mbpp_plus(row, config)
        elif config.key == "gsm8k":
            example = _gsm8k(row, config, index)
        elif config.key in {"aime24", "aime25"}:
            example = _aime(row, config, index)
        elif config.key == "strategyqa":
            example = _strategyqa(row, config, index)
        else:
            raise ValueError(f"unsupported benchmark task: {config.key}")
        examples.append(example)
    return tuple(examples)
