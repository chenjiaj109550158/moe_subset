"""Deterministic benchmark answer extraction and code execution."""

from __future__ import annotations

import math
import os
import re
import resource
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import cast

from pseudoroute.benchmark.tasks import BenchmarkExample


@dataclass(frozen=True)
class ScoreResult:
    correct: bool
    parsed_answer: str
    detail: str


def _last_boxed(text: str) -> str | None:
    start = max(text.rfind(r"\boxed"), text.rfind(r"\fbox"))
    if start < 0:
        return None
    brace = text.find("{", start)
    if brace < 0:
        match = re.search(r"\\boxed\s+([^\s$]+)", text[start:])
        return match.group(1) if match else None
    depth = 0
    for index in range(brace, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[brace + 1 : index]
    return None


def _normalize_math(text: str) -> str:
    value = text.replace("\n", "").replace(r"\!", "").replace(r"\\", "\\")
    value = value.replace("tfrac", "frac").replace("dfrac", "frac")
    value = value.replace(r"\left", "").replace(r"\right", "")
    value = value.replace(r"^{\circ}", "").replace(r"^\circ", "")
    value = value.replace(r"\$", "").replace(r"\%", "").replace("%", "")
    value = re.sub(r"\\text\{\s*[^}]*\}", "", value)
    if len(value.split("=")) == 2 and len(value.split("=")[0]) <= 2:
        value = value.split("=")[1]
    return value.replace(" ", "").strip(".$")


def _score_aime(text: str, target: str) -> ScoreResult:
    boxed = _last_boxed(text)
    if boxed is not None:
        answer = boxed
    else:
        dollars = [index for index, char in enumerate(text) if char == "$"]
        answer = text[dollars[0] + 1 : dollars[-1]] if len(dollars) > 1 else text
    parsed = _normalize_math(answer)
    expected = _normalize_math(target)
    return ScoreResult(parsed == expected, parsed, f"expected={expected}")


def _score_gsm8k(text: str, target: str) -> ScoreResult:
    number_pattern = r"-?\$?[0-9][0-9,]*(?:\.[0-9]+)?"
    boxed = _last_boxed(text)
    # LaTeX commonly renders thousands separators as ``{,}`` and escaped
    # currency as ``\$``. Normalize those before looking for a numeric answer.
    normalized = text.replace(r"{,}", ",").replace(r"\$", "$")
    normalized_boxed = boxed.replace(r"{,}", ",").replace(r"\$", "$") if boxed else ""
    boxed_numbers = re.findall(number_pattern, normalized_boxed)
    # Prefer the first number following the *last* explicit answer label. The
    # first number after a leading "Final Answer" heading is not necessarily the
    # answer when a verbose model subsequently shows its work, while the last
    # number in an answer sentence can be a unit conversion or explanation.
    answer_labels = list(
        re.finditer(
            r"(?:\*\*|__)?(?:final\s+)?answer(?:\*\*|__)?"
            r"\s*(?:is|:)(?:\*\*|__)?\s*",
            normalized,
            flags=re.IGNORECASE,
        )
    )
    answer_section = normalized[answer_labels[-1].end() :] if answer_labels else ""
    # Keep only the answer lead-in when the model follows it with a separately
    # styled explanation. This avoids treating worked-example values as answers.
    answer_intro = re.split(
        r"(?:\r?\n[ \t]*){2,}(?=(?:\*\*|__))",
        answer_section,
        maxsplit=1,
    )[0]
    answer_intro = re.split(
        r"(?:\*\*|__)(?:explanation|reasoning)(?:\*\*|__)",
        answer_intro,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    answer_numbers = re.findall(number_pattern, answer_intro)
    answer_emphasized = [
        numbers
        for section in re.findall(
            r"(?:\*\*|__)(.*?)(?:\*\*|__)",
            answer_intro,
            flags=re.DOTALL,
        )
        if (numbers := re.findall(number_pattern, section))
    ]
    emphasized_numbers = [
        numbers
        for section in re.findall(
            r"(?:\*\*|__)(.*?)(?:\*\*|__)",
            normalized,
            flags=re.DOTALL,
        )
        if "step" not in section.casefold()
        if (numbers := re.findall(number_pattern, section))
    ]
    candidates = re.findall(number_pattern, normalized)
    raw = (
        boxed_numbers[-1]
        if boxed_numbers
        else answer_emphasized[0][0]
        if answer_emphasized
        else answer_numbers[0]
        if answer_numbers
        else emphasized_numbers[-1][0]
        if emphasized_numbers
        else candidates[-1]
        if candidates
        else ""
    )
    parsed = raw.replace("$", "").replace(",", "").rstrip(".")
    expected = target.replace("$", "").replace(",", "").rstrip(".")
    try:
        correct = Decimal(parsed) == Decimal(expected)
    except InvalidOperation:
        correct = False
    return ScoreResult(correct, parsed, f"expected={expected}")


def _score_strategyqa(text: str, target: str) -> ScoreResult:
    candidates = re.findall(r"\b(yes|no|true|false)\b", text.lower())
    parsed = candidates[-1] if candidates else ""
    parsed = {"true": "yes", "false": "no"}.get(parsed, parsed)
    return ScoreResult(parsed == target, parsed, f"expected={target}")


def _extract_code(example: BenchmarkExample, text: str) -> str:
    if example.assistant_prefix is None:
        tagged_blocks = cast(
            list[tuple[str, str]],
            re.findall(
                r"```[ \t]*(?:(python(?:3)?|py)(?=\s)[ \t]*)?(?:\r?\n)?(.*?)```",
                text,
                flags=re.DOTALL | re.IGNORECASE,
            ),
        )
        blocks = [block for _, block in tagged_blocks]
        if example.task == "humaneval":
            entry_points = (str(example.row["entry_point"]),)
        else:
            entry_points = tuple(
                re.findall(r"\bdef\s+([A-Za-z_]\w*)\s*\(", str(example.row.get("code", "")))
            )
        complete = next(
            (
                block
                for block in blocks
                if any(
                    re.search(rf"\bdef\s+{re.escape(entry_point)}\b", block)
                    for entry_point in entry_points
                )
            ),
            None,
        )
        if complete is not None:
            return complete
        python_block = next(
            (
                block
                for language, block in tagged_blocks
                if language.casefold() in {"python", "python3", "py"}
            ),
            None,
        )
        if python_block is not None:
            return python_block
        if blocks:
            return blocks[0]
    before_fence = text.split("```", 1)[0]
    if before_fence.strip():
        if example.task == "humaneval":
            return str(example.row["prompt"]) + before_fence
        return before_fence
    fenced_source = text
    if text.lstrip().startswith("```"):
        # The prompt already opened a fence. Some chat models close it immediately,
        # then emit a complete replacement in a second fenced block.
        fenced_source = text.lstrip()[3:]
    blocks = cast(
        list[str],
        re.findall(
            r"```(?:python)?\s*(.*?)```",
            fenced_source,
            flags=re.DOTALL | re.IGNORECASE,
        ),
    )
    if example.task == "humaneval":
        entry_point = str(example.row["entry_point"])
        complete = next(
            (block for block in blocks if re.search(rf"\bdef\s+{re.escape(entry_point)}\b", block)),
            None,
        )
        if complete is not None:
            return complete
        return str(example.row["prompt"]) + before_fence
    if blocks:
        return blocks[0]
    return before_fence


def _limit_code_process() -> None:
    resource.setrlimit(resource.RLIMIT_CPU, (12, 12))
    resource.setrlimit(resource.RLIMIT_AS, (2 * 1024**3, 2 * 1024**3))
    resource.setrlimit(resource.RLIMIT_FSIZE, (16 * 1024**2, 16 * 1024**2))
    resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))


_SAFETY_PREAMBLE = """
import asyncio
import doctest
import os
import shutil
import socket
import ssl
import subprocess

def _blocked(*args, **kwargs):
    raise PermissionError("operation disabled by benchmark harness")

os.system = _blocked
os.remove = _blocked
os.unlink = _blocked
os.rmdir = _blocked
os.removedirs = _blocked
os.rename = _blocked
os.replace = _blocked
shutil.rmtree = _blocked
shutil.move = _blocked
subprocess.Popen = _blocked
subprocess.call = _blocked
subprocess.run = _blocked
socket.socket = _blocked
"""


def _humaneval_prompt_prelude(example: BenchmarkExample) -> str:
    """Retain helper definitions that precede the target HumanEval function."""
    if example.task != "humaneval":
        return ""
    prompt = str(example.row.get("prompt", ""))
    entry_point = str(example.row.get("entry_point", ""))
    target = re.search(
        rf"(?m)^(?:async[ ]+)?def[ ]+{re.escape(entry_point)}[ ]*[(]",
        prompt,
    )
    return prompt[: target.start()] if target is not None else ""


def _score_code(example: BenchmarkExample, text: str) -> ScoreResult:
    code = _extract_code(example, text)
    if not code.strip():
        return ScoreResult(False, "", "empty code")
    # Compile the candidate as a separate source unit. This keeps a generated
    # ``from __future__`` legal even though the safety preamble must execute
    # first, and prevents candidate-only ``if __name__ == "__main__"`` blocks
    # from running as part of the benchmark.
    prelude = _humaneval_prompt_prelude(example)
    program = (
        f"{_SAFETY_PREAMBLE}\n"
        f"__name__ = '__candidate__'\n"
        f"exec(compile({prelude!r}, '<prompt-prelude>', 'exec'), globals())\n"
        f"exec(compile({code!r}, '<candidate>', 'exec'), globals())\n"
        f"{example.target}\n"
    )
    with tempfile.TemporaryDirectory(prefix="pseudoroute-code-") as directory:
        script = Path(directory) / "candidate.py"
        script.write_text(program, encoding="utf-8")
        try:
            result = subprocess.run(
                [sys.executable, "-I", str(script)],
                cwd=directory,
                env={"PATH": os.environ.get("PATH", ""), "PYTHONHASHSEED": "0"},
                capture_output=True,
                text=True,
                timeout=20,
                preexec_fn=_limit_code_process,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return ScoreResult(False, code, "timeout")
    detail = f"returncode={result.returncode}"
    if result.stderr:
        detail += f"; stderr={result.stderr[-500:]}"
    return ScoreResult(result.returncode == 0, code, detail)


def score_response(example: BenchmarkExample, text: str) -> ScoreResult:
    if "assistantfinal" in text:
        text = text.rsplit("assistantfinal", 1)[-1]
    elif text.startswith("analysis"):
        text = ""
    if example.task in {"humaneval", "mbpp_plus"}:
        return _score_code(example, text)
    if example.task == "gsm8k":
        return _score_gsm8k(text, example.target)
    if example.task in {"aime24", "aime25"}:
        return _score_aime(text, example.target)
    if example.task == "strategyqa":
        return _score_strategyqa(text, example.target)
    raise ValueError(f"unsupported benchmark task: {example.task}")


def wilson_interval(
    successes: int, total: int, z: float = 1.959963984540054
) -> tuple[float, float]:
    if total <= 0:
        raise ValueError("total must be positive")
    proportion = successes / total
    denominator = 1 + z * z / total
    centre = proportion + z * z / (2 * total)
    radius = z * math.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total**2))
    return (centre - radius) / denominator, (centre + radius) / denominator
