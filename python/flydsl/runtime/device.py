# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

import functools
import os
import subprocess
from typing import Optional

_ROCM_AGENT_TIMEOUT_S = int(os.environ.get("FLYDSL_ROCM_AGENT_TIMEOUT", "300"))


def _rocm_agent_names(timeout: int, *, visible_names: bool = False) -> list[str]:
    # The default output has one architecture per physical agent, so it is used
    # for counting. -name goes through rocminfo/HSA and therefore respects
    # runtime visibility, which is required for architecture detection.
    command = ["rocm_agent_enumerator"]
    if visible_names:
        command.append("-name")
    try:
        out = subprocess.check_output(
            command,
            text=True,
            timeout=timeout,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.SubprocessError, UnicodeError):
        return []
    return [name for line in out.splitlines() if (name := line.strip()).startswith("gfx") and name != "gfx000"]


def _arch_from_rocm_agent_enumerator() -> Optional[str]:
    """Query the first HSA-visible ROCm GPU architecture."""
    names = _rocm_agent_names(_ROCM_AGENT_TIMEOUT_S, visible_names=True)
    return names[0] if names else None


def _arch_from_rocminfo() -> Optional[str]:
    try:
        out = subprocess.check_output(
            ["rocminfo"],
            text=True,
            timeout=_ROCM_AGENT_TIMEOUT_S,
            stderr=subprocess.DEVNULL,
        )
        for line in out.splitlines():
            _, _, name = line.partition("Name:")
            name = name.strip()
            if name.startswith("gfx") and name != "gfx000":
                return name
    except (OSError, subprocess.SubprocessError, UnicodeError):
        pass
    return None


@functools.lru_cache(maxsize=None)
def _arch_from_hardware() -> str:
    """Cached hardware detection."""
    arch = _arch_from_rocm_agent_enumerator() or _arch_from_rocminfo()
    if arch:
        return arch.split(":", 1)[0]
    raise RuntimeError(
        "Unable to detect a ROCm GPU architecture with rocm_agent_enumerator or rocminfo. "
        "Set FLYDSL_GPU_ARCH to the intended gfx target when hardware detection is unavailable."
    )


def get_rocm_arch() -> str:
    """ROCm GPU arch string, always lower-cased (e.g. 'gfx942').

    Lower-casing happens here so every caller can compare against lower-case
    literals without normalising first; ROCm itself only ever emits lower-case
    names, so this only affects hand-set environment overrides.
    Raises RuntimeError if neither an override nor hardware detection supplies
    a target; unavailable hardware must not silently select another GPU's ISA.
    """
    env = os.environ.get("FLYDSL_GPU_ARCH") or os.environ.get("HSA_OVERRIDE_GFX_VERSION")
    if env:
        env = env.lower()
        if env.startswith("gfx"):
            return env
        if env.count(".") == 2:
            parts = env.split(".")
            return f"gfx{parts[0]}{parts[1]}{parts[2]}"

    return _arch_from_hardware().lower()


@functools.lru_cache(maxsize=None)
def get_rocm_device_count() -> int:
    """Best-effort ROCm visible GPU count via ``rocm_agent_enumerator`` (standard ROCm tool).

    Returns 0 when the tool is unavailable or no discrete GPU agents are reported.
    """
    return len(_rocm_agent_names(5))


def is_rdna_arch(arch: Optional[str] = None) -> bool:
    """Check if architecture is RDNA-based (gfx10, gfx11, or gfx120).

    This is the single source of truth for CDNA vs RDNA classification.
    RDNA architectures use wave32 and have different buffer descriptor flags.
    CDNA5 gfx1250 is also wave32, but is intentionally not classified as RDNA.

    If arch is None, the current GPU arch is auto-detected.
    """
    if arch is None:
        arch = get_rocm_arch()
    if not arch:
        return False
    arch = arch.lower()
    if arch.startswith("gfx10") or arch.startswith("gfx11"):
        return True
    if arch.startswith("gfx120"):
        return True
    return False


def get_warp_size(arch: Optional[str] = None) -> int:
    """Lanes per warp for an architecture.

    If arch is None, the current GPU arch is auto-detected.
    """
    if arch is None:
        arch = get_rocm_arch()
    if not arch:
        return 64
    arch = arch.lower()
    if arch.startswith("gfx10") or arch.startswith("gfx11") or arch.startswith("gfx12"):
        return 32
    return 64


# LDS bytes available per CU / WGP, keyed by architecture.
SMEM_CAPACITY_MAP = {
    "gfx942": 65536,
    "gfx950": 163840,
    "gfx1100": 65536,
    "gfx1151": 65536,
    "gfx1201": 65536,
    "gfx1250": 327680,
}


def check_smem_capacity(allocated_bytes: int, arch: str = None):
    """Raise if the requested shared-memory bytes exceed the device LDS capacity.

    The check is skipped when ``arch`` is None or not in the capacity map.
    """
    if arch is None:
        return
    limit = SMEM_CAPACITY_MAP.get(arch)
    if limit is not None and allocated_bytes > limit:
        raise RuntimeError(
            f"Shared Memory Overflow: Requested {allocated_bytes} bytes, " f"but device {arch} limit is {limit} bytes."
        )
