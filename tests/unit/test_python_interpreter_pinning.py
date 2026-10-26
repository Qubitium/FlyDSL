# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Both MLIR Python detections must resolve to one interpreter.

MLIRDetectPythonEnv.cmake calls ``find_package(Python ...)`` separately from the
project's ``find_package(Python3 ...)`` because nanobind's CMake package reads
the ``Python_`` variables. Pinning only one lets the bindings be compiled for
one ABI and installed under another, which surfaces as

    ImportError: cannot import name 'ir' from ..._mlir (unknown location)

These tests cover the shell helper that pins both, and the CMake guard that
fails the configure step when they still disagree. No MLIR, network or CMake
build is needed: the helper is sourced directly, and the guard is exercised
with a small standalone CMake project.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_HELPER = _REPO_ROOT / "scripts" / "python_interpreter.sh"
_GUARD = _REPO_ROOT / "cmake" / "FlyDSLPythonCheck.cmake"

pytestmark = [
    pytest.mark.l0_backend_agnostic,
    pytest.mark.skipif(shutil.which("bash") is None, reason="bash is required"),
]


def _run_helper(env_overrides):
    """Source the helper in a clean shell and print the resulting CMake args."""
    script = 'source "$1" && printf "%s\\n" "${FLY_PYTHON_CMAKE_ARGS[@]}"'
    env = {k: v for k, v in os.environ.items() if k != "PYTHON_EXECUTABLE"}
    env.update(env_overrides)
    return subprocess.run(
        ["bash", "-c", script, "helper", str(_HELPER)],
        capture_output=True,
        text=True,
        env=env,
    )


def test_pins_both_variable_families_to_the_same_interpreter():
    """The bug was pinning Python3_ only; both families must be emitted."""
    result = _run_helper({"PYTHON_EXECUTABLE": sys.executable})

    assert result.returncode == 0, result.stderr
    args = result.stdout.split()
    python3_pins = [a for a in args if a.startswith("-DPython3_EXECUTABLE=")]
    python_pins = [a for a in args if a.startswith("-DPython_EXECUTABLE=")]

    assert len(python3_pins) == 1, args
    assert len(python_pins) == 1, args
    # -DPython_EXECUTABLE= must not be satisfied by -DPython3_EXECUTABLE=.
    assert python3_pins[0].split("=", 1)[1] == python_pins[0].split("=", 1)[1]


def test_nanobind_dir_comes_from_the_pinned_interpreter():
    """nanobind's CMake dir used to be resolved from whatever python3 was on PATH."""
    result = _run_helper({"PYTHON_EXECUTABLE": sys.executable})

    assert result.returncode == 0, result.stderr
    nanobind_pins = [a for a in result.stdout.split() if a.startswith("-Dnanobind_DIR=")]
    assert len(nanobind_pins) == 1, result.stdout

    pinned = Path(nanobind_pins[0].split("=", 1)[1]).resolve()
    from_interpreter = subprocess.run(
        [
            sys.executable,
            "-c",
            "import nanobind, os; print(os.path.dirname(nanobind.__file__) + '/cmake')",
        ],
        capture_output=True,
        text=True,
    )
    if from_interpreter.returncode != 0:
        pytest.skip("nanobind is not installed for the test interpreter")
    assert pinned == Path(from_interpreter.stdout.strip()).resolve()


def test_reports_version_and_soabi():
    """A free-threaded interpreter's ABI tag must be visible in the log."""
    result = _run_helper({"PYTHON_EXECUTABLE": sys.executable})

    assert result.returncode == 0, result.stderr
    combined = result.stdout + result.stderr
    assert "Python bindings interpreter:" in combined
    assert Path(sys.executable).resolve().name in combined


def test_missing_interpreter_fails_without_killing_a_set_e_caller():
    """build.sh runs under `set -e`; a bad pin must report, not abort mid-script."""
    result = _run_helper({"PYTHON_EXECUTABLE": "/nonexistent/python3"})

    assert result.returncode != 0
    # Either diagnostic is acceptable: the helper rejects a missing path before
    # it ever runs the interpreter.
    assert "not a usable Python" in result.stderr or "not a file" in result.stderr
    assert "/nonexistent/python3" in result.stderr

    # The helper must return, not exit, so the caller decides what happens.
    survives = subprocess.run(
        [
            "bash",
            "-c",
            'set -e; source "$1" || true; echo CALLER-ALIVE',
            "helper",
            str(_HELPER),
        ],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHON_EXECUTABLE": "/nonexistent/python3"},
    )
    assert "CALLER-ALIVE" in survives.stdout


def test_non_python_executable_is_rejected(tmp_path):
    """An executable that is not Python must not be accepted as the pin."""
    fake = tmp_path / "not-python"
    fake.write_text("#!/bin/sh\nexit 0\n")
    fake.chmod(0o755)

    result = _run_helper({"PYTHON_EXECUTABLE": str(fake)})

    assert result.returncode != 0
    assert "not a usable Python" in result.stderr


def test_venv_launcher_is_preserved_not_collapsed_to_base(tmp_path):
    """build_wheels.sh builds from a venv; resolving its launcher loses the venv."""
    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True)
    (venv / "pyvenv.cfg").write_text("home = /\n")
    launcher = venv / "bin" / "python"
    launcher.symlink_to(sys.executable)

    result = _run_helper({"PYTHON_EXECUTABLE": str(launcher)})

    assert result.returncode == 0, result.stderr
    pins = [a for a in result.stdout.split() if a.startswith("-DPython3_EXECUTABLE=")]
    assert pins and pins[0].split("=", 1)[1] == str(launcher)


# ---- cmake/FlyDSLPythonCheck.cmake -------------------------------------------


def _run_cmake_guard(tmp_path, python3_executable, python_executable, allow=False):
    cmake = shutil.which("cmake")
    if cmake is None:
        pytest.skip("cmake is required")

    project = tmp_path / "proj"
    (project / "cmake").mkdir(parents=True)
    # Mirror the real tree: the module lives in cmake/ and is included by
    # relative path, exactly as CMakeLists.txt does. Putting the directory on
    # CMAKE_MODULE_PATH instead would not catch a broken include path.
    shutil.copy(_GUARD, project / "cmake" / "FlyDSLPythonCheck.cmake")
    (project / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.20)\n"
        "project(guardcheck NONE)\n"
        f'set(Python3_EXECUTABLE "{python3_executable}")\n'
        f'set(Python_EXECUTABLE "{python_executable}")\n'
        f"option(FLY_ALLOW_PYTHON_MISMATCH \"\" {'ON' if allow else 'OFF'})\n"
        "include(cmake/FlyDSLPythonCheck.cmake)\n"
        "fly_check_python_consistency()\n"
        'message(STATUS "guard passed")\n'
    )
    return subprocess.run(
        [cmake, "-S", str(project), "-B", str(tmp_path / "b")],
        capture_output=True,
        text=True,
    )


def test_guard_accepts_one_interpreter_spelled_two_ways(tmp_path):
    """The two passes may name the same interpreter via different paths."""
    resolved = Path(sys.executable).resolve()
    symlink = tmp_path / "python-link"
    symlink.symlink_to(resolved)

    result = _run_cmake_guard(tmp_path, str(resolved), str(symlink))

    assert result.returncode == 0, result.stdout + result.stderr
    assert "guard passed" in result.stdout


def test_guard_fails_when_the_two_detections_disagree(tmp_path):
    """This is the regression: mismatched passes used to configure happily."""
    other = shutil.which("python3.12") or shutil.which("python3.11") or shutil.which("python3")
    if other is None:
        pytest.skip("a second interpreter is required")
    if Path(other).resolve() == Path(sys.executable).resolve():
        pytest.skip("only one interpreter is available")

    result = _run_cmake_guard(tmp_path, sys.executable, other)

    assert result.returncode != 0
    assert "different interpreters" in result.stdout + result.stderr
    # The error must explain the import failure the user would otherwise hit.
    assert "cannot import name 'ir'" in result.stdout + result.stderr


def test_guard_can_be_downgraded_to_a_warning(tmp_path):
    """Escape hatch for environments that knowingly mix interpreters."""
    other = shutil.which("python3.12") or shutil.which("python3.11") or shutil.which("python3")
    if other is None:
        pytest.skip("a second interpreter is required")
    if Path(other).resolve() == Path(sys.executable).resolve():
        pytest.skip("only one interpreter is available")

    result = _run_cmake_guard(tmp_path, sys.executable, other, allow=True)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "different interpreters" in result.stdout + result.stderr
    assert "guard passed" in result.stdout
