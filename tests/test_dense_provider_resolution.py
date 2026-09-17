"""Tests for ant.retrieval.dense._resolve_onnx_providers -- the auto GPU/CPU
provider detection that lets the same DenseEmbedder code path run unmodified
on a GPU cluster node and a CPU-only machine.
"""
from __future__ import annotations

import pytest

from ant.retrieval import dense as dense_module


class _FakeOnnxRuntime:
    def __init__(self, available: list[str]) -> None:
        self._available = available

    def get_available_providers(self) -> list[str]:
        return self._available


def _patch_onnxruntime(monkeypatch: pytest.MonkeyPatch, available: list[str]) -> None:
    import sys

    monkeypatch.setitem(sys.modules, "onnxruntime", _FakeOnnxRuntime(available))


def test_cpu_only_install_resolves_to_cpu_provider_alone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ANT_EMBEDDING_PROVIDER", raising=False)
    _patch_onnxruntime(monkeypatch, ["CPUExecutionProvider"])
    providers = dense_module._resolve_onnx_providers()
    assert providers == [("CPUExecutionProvider", {"arena_extend_strategy": "kSameAsRequested"})]


def test_auto_prefers_cuda_when_available_with_cpu_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ANT_EMBEDDING_PROVIDER", raising=False)
    _patch_onnxruntime(monkeypatch, ["CUDAExecutionProvider", "CPUExecutionProvider"])
    providers = dense_module._resolve_onnx_providers()
    assert providers[0] == "CUDAExecutionProvider"
    assert providers[-1] == ("CPUExecutionProvider", {"arena_extend_strategy": "kSameAsRequested"})


def test_auto_respects_gpu_priority_order(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANT_EMBEDDING_PROVIDER", raising=False)
    # ROCm listed as available alongside CUDA -- CUDA must still win (it's
    # first in _GPU_PROVIDER_PRIORITY), order in `available` must not matter.
    _patch_onnxruntime(
        monkeypatch, ["ROCMExecutionProvider", "CPUExecutionProvider", "CUDAExecutionProvider"]
    )
    providers = dense_module._resolve_onnx_providers()
    assert providers[0] == "CUDAExecutionProvider"


def test_override_cpu_forces_cpu_even_when_gpu_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANT_EMBEDDING_PROVIDER", "cpu")
    _patch_onnxruntime(monkeypatch, ["CUDAExecutionProvider", "CPUExecutionProvider"])
    providers = dense_module._resolve_onnx_providers()
    assert providers == [("CPUExecutionProvider", {"arena_extend_strategy": "kSameAsRequested"})]


def test_override_explicit_provider_name_is_tried_first(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANT_EMBEDDING_PROVIDER", "TensorrtExecutionProvider")
    _patch_onnxruntime(monkeypatch, ["CPUExecutionProvider"])
    providers = dense_module._resolve_onnx_providers()
    assert providers[0] == "TensorrtExecutionProvider"
    assert providers[-1] == ("CPUExecutionProvider", {"arena_extend_strategy": "kSameAsRequested"})


def test_no_gpu_providers_available_falls_back_cleanly_with_auto(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANT_EMBEDDING_PROVIDER", "auto")
    _patch_onnxruntime(monkeypatch, ["CPUExecutionProvider"])
    providers = dense_module._resolve_onnx_providers()
    assert providers == [("CPUExecutionProvider", {"arena_extend_strategy": "kSameAsRequested"})]
