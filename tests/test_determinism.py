import pytest
import torch

from pseudoroute.config import resolve_device


def test_auto_device_resolves_to_available_backend() -> None:
    expected = "cuda" if torch.cuda.is_available() else "cpu"
    assert resolve_device("auto") == expected


def test_explicit_cpu_device() -> None:
    assert resolve_device("cpu") == "cpu"


def test_explicit_unavailable_cuda_fails() -> None:
    if torch.cuda.is_available():
        pytest.skip("CUDA is available")
    with pytest.raises(RuntimeError, match="not available"):
        resolve_device("cuda")
