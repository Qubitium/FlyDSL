# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

import subprocess

import pytest

from flydsl.runtime import device

pytestmark = [pytest.mark.l0_backend_agnostic]


@pytest.fixture(autouse=True)
def _clear_arch_caches():
    device._arch_from_hardware.cache_clear()
    device.get_rocm_device_count.cache_clear()
    yield
    device._arch_from_hardware.cache_clear()
    device.get_rocm_device_count.cache_clear()


def test_arch_detection_uses_visible_device_query(monkeypatch):
    calls = []

    def check_output(command, **kwargs):
        calls.append((command, kwargs))
        return "gfx000\ngfx950:sramecc+:xnack-\n"

    monkeypatch.setattr(subprocess, "check_output", check_output)

    assert device._arch_from_rocm_agent_enumerator() == "gfx950:sramecc+:xnack-"
    assert calls[0][0] == ["rocm_agent_enumerator", "-name"]


def test_arch_detection_ignores_hidden_heterogeneous_gpu(monkeypatch):
    """KFD counting order must not choose the architecture of a hidden GPU."""

    def check_output(command, **_kwargs):
        if command == ["rocm_agent_enumerator"]:
            return "gfx1100\ngfx950\n"
        assert command == ["rocm_agent_enumerator", "-name"]
        return "gfx950:sramecc+:xnack-\ngfx9-4-generic:sramecc+:xnack-\n"

    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "1")
    monkeypatch.setattr(subprocess, "check_output", check_output)

    assert device._arch_from_hardware() == "gfx950"


@pytest.mark.parametrize("failure", ["missing", "timeout", "exit", "empty", "cpu-only"])
def test_hardware_detection_falls_back_to_rocminfo(monkeypatch, failure):
    def check_output(command, **_kwargs):
        if command == ["rocm_agent_enumerator", "-name"]:
            if failure == "missing":
                raise FileNotFoundError(command[0])
            if failure == "timeout":
                raise subprocess.TimeoutExpired(command, 1)
            if failure == "exit":
                raise subprocess.CalledProcessError(1, command)
            return "gfx000\n" if failure == "cpu-only" else ""
        assert command == ["rocminfo"]
        return "  Name:                    gfx000\n  Name:                    gfx950:sramecc+:xnack-\n"

    monkeypatch.setattr(subprocess, "check_output", check_output)

    assert device._arch_from_hardware() == "gfx950"


@pytest.mark.parametrize("failure", ["missing", "timeout", "exit", "empty"])
def test_hardware_detection_does_not_default_to_gfx942(monkeypatch, failure):
    def check_output(command, **_kwargs):
        if failure == "missing":
            raise FileNotFoundError(command[0])
        if failure == "timeout":
            raise subprocess.TimeoutExpired(command, 1)
        if failure == "exit":
            raise subprocess.CalledProcessError(1, command)
        return ""

    monkeypatch.setattr(subprocess, "check_output", check_output)

    with pytest.raises(RuntimeError, match="FLYDSL_GPU_ARCH"):
        device._arch_from_hardware()


@pytest.mark.parametrize(
    ("name", "value", "expected"),
    [
        ("FLYDSL_GPU_ARCH", "GFX950", "gfx950"),
        ("HSA_OVERRIDE_GFX_VERSION", "9.5.0", "gfx950"),
    ],
)
def test_environment_override_bypasses_hardware_detection(monkeypatch, name, value, expected):
    monkeypatch.delenv("FLYDSL_GPU_ARCH", raising=False)
    monkeypatch.delenv("HSA_OVERRIDE_GFX_VERSION", raising=False)
    monkeypatch.setenv(name, value)
    monkeypatch.setattr(
        device,
        "_arch_from_hardware",
        lambda: pytest.fail("hardware detection should not run"),
    )

    assert device.get_rocm_arch() == expected


def test_device_count_uses_default_agent_enumerator_invocation(monkeypatch):
    calls = []

    def check_output(command, **kwargs):
        calls.append((command, kwargs))
        return "gfx000\ngfx950\ngfx950\n"

    monkeypatch.setattr(subprocess, "check_output", check_output)

    assert device.get_rocm_device_count() == 2
    assert calls[0][0] == ["rocm_agent_enumerator"]


def test_device_count_counts_gpus_not_supported_isas(monkeypatch):
    """Two identical GPUs must count as two, even when each exposes two ISAs."""

    def check_output(command, **_kwargs):
        if command == ["rocm_agent_enumerator", "-name"]:
            return "gfx950:sramecc+:xnack-\ngfx9-4-generic:sramecc+:xnack-\n" * 2
        assert command == ["rocm_agent_enumerator"]
        return "gfx950\ngfx950\n"

    monkeypatch.setattr(subprocess, "check_output", check_output)

    assert device.get_rocm_device_count() == 2


@pytest.mark.parametrize("failure", ["missing", "timeout"])
def test_device_count_remains_best_effort(monkeypatch, failure):
    def check_output(command, **_kwargs):
        if failure == "missing":
            raise FileNotFoundError(command[0])
        raise subprocess.TimeoutExpired(command, 5)

    monkeypatch.setattr(subprocess, "check_output", check_output)

    assert device.get_rocm_device_count() == 0
