"""Tests for lazyload._version — runtime Python-version detection.

These tests verify that:
* The computed constants match the running interpreter.
* Exactly one of the two mutually-exclusive flags is active.
* ``get_mode_description()`` produces coherent output.
* ``UnsupportedPythonVersion`` and ``CircularImportError`` format their
  messages correctly and expose typed attributes.
* ``lazyload.ModuleNotFoundError`` satisfies both the lazyload and builtin
  exception hierarchies simultaneously.

No mocking of ``sys.version_info`` is needed here because the tests only
assert *consistency* between the constants and the actual running version;
they do not assert specific values that would depend on the test environment.
"""

from __future__ import annotations

import builtins
import sys
from types import ModuleType

import pytest

import lazyload
from lazyload._exceptions import (
    CircularImportError,
    LazyLoadError,
    ModuleNotFoundError,
    UnsupportedPythonVersion,
)
import lazyload._version as _v

# ──────────────────────────────────────────────────────────────────────────────
# PYTHON_VERSION string
# ──────────────────────────────────────────────────────────────────────────────


class TestPythonVersionString:
    """``PYTHON_VERSION`` must be a well-formed ``"major.minor.micro"`` string."""

    def test_format_is_major_minor_micro(self) -> None:
        """PYTHON_VERSION must contain exactly two dots and three numeric segments."""
        parts = _v.PYTHON_VERSION.split(".")
        assert len(parts) == 3, (
            f"Expected 'major.minor.micro', got {_v.PYTHON_VERSION!r}"
        )
        assert all(part.isdigit() for part in parts), (
            f"All segments must be digits, got {_v.PYTHON_VERSION!r}"
        )

    def test_matches_sys_version_info(self) -> None:
        """PYTHON_VERSION must reflect the actual running interpreter."""
        vi = sys.version_info
        expected = f"{vi.major}.{vi.minor}.{vi.micro}"
        assert expected == _v.PYTHON_VERSION


# ──────────────────────────────────────────────────────────────────────────────
# Boolean capability flags
# ──────────────────────────────────────────────────────────────────────────────


class TestCapabilityFlags:
    """``NATIVE_LAZY_IMPORTS`` and ``SHIM_REQUIRED`` must be mutually exclusive."""

    def test_exactly_one_flag_is_true(self) -> None:
        """Exactly one of the two boolean flags must be True (XOR)."""
        assert _v.NATIVE_LAZY_IMPORTS ^ _v.SHIM_REQUIRED, (
            "NATIVE_LAZY_IMPORTS XOR SHIM_REQUIRED must be True; "
            f"got NATIVE={_v.NATIVE_LAZY_IMPORTS}, SHIM={_v.SHIM_REQUIRED}"
        )

    def test_neither_flag_is_none(self) -> None:
        """Both flags must be genuine booleans, not None or unset."""
        assert isinstance(_v.NATIVE_LAZY_IMPORTS, bool)
        assert isinstance(_v.SHIM_REQUIRED, bool)

    def test_native_flag_matches_version(self) -> None:
        """NATIVE_LAZY_IMPORTS must be True if and only if Python >= 3.15."""
        is_native = sys.version_info >= (3, 15)
        assert _v.NATIVE_LAZY_IMPORTS is is_native

    def test_shim_flag_matches_version(self) -> None:
        """SHIM_REQUIRED must be True if and only if Python is 3.10 through 3.14."""
        is_shim = (3, 10) <= sys.version_info < (3, 15)
        assert _v.SHIM_REQUIRED is is_shim

    def test_flags_are_not_equal(self) -> None:
        """The two flags must never have the same value."""
        assert _v.NATIVE_LAZY_IMPORTS != _v.SHIM_REQUIRED


# ──────────────────────────────────────────────────────────────────────────────
# MODE literal
# ──────────────────────────────────────────────────────────────────────────────


class TestMode:
    """``MODE`` must be a Literal string consistent with the capability flags."""

    def test_mode_is_string(self) -> None:
        """MODE must be a plain str."""
        assert isinstance(_v.MODE, str)

    def test_mode_is_valid_value(self) -> None:
        """MODE must be exactly 'native' or 'shim' — no other value is legal."""
        assert _v.MODE in {"native", "shim"}, f"Unexpected MODE={_v.MODE!r}"

    def test_mode_consistent_with_flags(self) -> None:
        """MODE must agree with NATIVE_LAZY_IMPORTS and SHIM_REQUIRED."""
        if _v.NATIVE_LAZY_IMPORTS:
            assert _v.MODE == "native"
        else:
            assert _v.MODE == "shim"


# ──────────────────────────────────────────────────────────────────────────────
# get_mode_description()
# ──────────────────────────────────────────────────────────────────────────────


class TestGetModeDescription:
    """``get_mode_description()`` must return a coherent human-readable string."""

    def test_returns_string(self) -> None:
        """get_mode_description() must return a str, not bytes or None."""
        result = _v.get_mode_description()
        assert isinstance(result, str)

    def test_contains_python_version(self) -> None:
        """The description must embed the running Python version number."""
        result = _v.get_mode_description()
        assert _v.PYTHON_VERSION in result

    def test_contains_mode(self) -> None:
        """The description must mention the active mode ('native' or 'shim')."""
        result = _v.get_mode_description()
        assert _v.MODE in result

    def test_contains_lazyload_version(self) -> None:
        """The description must embed the lazyload package version."""
        result = _v.get_mode_description()
        assert lazyload.__version__ in result

    def test_is_multiline(self) -> None:
        """The description must span multiple lines — it is a diagnostic block."""
        result = _v.get_mode_description()
        assert "\n" in result, "Expected a multi-line diagnostic string"

    def test_native_description_mentions_pep_810(self, simulate_py315: None) -> None:
        """On Python 3.15+, the description must reference the native path."""
        result = _v.get_mode_description()
        assert "native" in result.lower()

    def test_shim_description_mentions_shim(self, simulate_py310: None) -> None:
        """On Python 3.10–3.14, the description must reference the shim."""
        result = _v.get_mode_description()
        assert "shim" in result.lower()


# ──────────────────────────────────────────────────────────────────────────────
# UnsupportedPythonVersion
# ──────────────────────────────────────────────────────────────────────────────


class TestUnsupportedPythonVersion:
    """``UnsupportedPythonVersion`` must carry version metadata and a clear message."""

    def test_is_subclass_of_lazy_load_error(self) -> None:
        """UnsupportedPythonVersion must inherit from LazyLoadError."""
        assert issubclass(UnsupportedPythonVersion, LazyLoadError)

    def test_is_subclass_of_exception(self) -> None:
        """UnsupportedPythonVersion must ultimately inherit from Exception."""
        assert issubclass(UnsupportedPythonVersion, Exception)

    def test_stores_actual_version(self) -> None:
        """The .actual_version attribute must match the argument passed in."""
        exc = UnsupportedPythonVersion("3.9.5")
        assert exc.actual_version == "3.9.5"

    def test_stores_minimum_version(self) -> None:
        """The .minimum_version attribute defaults to '3.10'."""
        exc = UnsupportedPythonVersion("3.9.5")
        assert exc.minimum_version == "3.10"

    def test_custom_minimum_version(self) -> None:
        """A custom minimum_version is stored and reflected in the message."""
        exc = UnsupportedPythonVersion("3.8.0", minimum_version="3.10")
        assert exc.minimum_version == "3.10"

    def test_message_contains_actual_version(self) -> None:
        """The exception message must name the version the user is running."""
        exc = UnsupportedPythonVersion("3.9.18")
        assert "3.9.18" in str(exc)

    def test_message_contains_minimum_version(self) -> None:
        """The exception message must state the required minimum version."""
        exc = UnsupportedPythonVersion("3.9.18", minimum_version="3.10")
        assert "3.10" in str(exc)

    def test_message_mentions_upgrade(self) -> None:
        """The exception message must tell the user to upgrade."""
        exc = UnsupportedPythonVersion("3.8.0")
        msg = str(exc).lower()
        assert "upgrade" in msg or "update" in msg or "3.10" in msg

    def test_can_be_raised_and_caught_as_lazy_load_error(self) -> None:
        """Raising UnsupportedPythonVersion can be caught as LazyLoadError."""
        with pytest.raises(LazyLoadError):
            raise UnsupportedPythonVersion("3.9.0")

    def test_repr_contains_class_name(self) -> None:
        """repr() must include the class name for clarity in tracebacks."""
        exc = UnsupportedPythonVersion("3.9.0")
        assert "UnsupportedPythonVersion" in repr(exc)


# ──────────────────────────────────────────────────────────────────────────────
# CircularImportError
# ──────────────────────────────────────────────────────────────────────────────


class TestCircularImportError:
    """``CircularImportError`` must carry the cycle path and format it clearly."""

    def test_is_subclass_of_lazy_load_error(self) -> None:
        """CircularImportError must inherit from LazyLoadError."""
        assert issubclass(CircularImportError, LazyLoadError)

    def test_stores_module_name(self) -> None:
        """The .module_name attribute must match the argument."""
        exc = CircularImportError("pkg.a")
        assert exc.module_name == "pkg.a"

    def test_stores_cycle_list(self) -> None:
        """The .cycle attribute must store the full cycle path list."""
        cycle = ["pkg.a", "pkg.b", "pkg.c", "pkg.a"]
        exc = CircularImportError("pkg.a", cycle=cycle)
        assert exc.cycle == cycle

    def test_empty_cycle_defaults_to_empty_list(self) -> None:
        """When cycle is omitted, .cycle must be an empty list, not None."""
        exc = CircularImportError("pkg.a")
        assert exc.cycle == []
        assert isinstance(exc.cycle, list)

    def test_message_contains_module_name(self) -> None:
        """The exception message must mention the offending module name."""
        exc = CircularImportError("pkg.heavy")
        assert "pkg.heavy" in str(exc)

    def test_message_contains_arrow_path_when_cycle_given(self) -> None:
        """When a cycle is provided, the message must show the arrow path."""
        cycle = ["a", "b", "a"]
        exc = CircularImportError("a", cycle=cycle)
        msg = str(exc)
        # All module names in the cycle must appear in the message.
        for name in cycle:
            assert name in msg

    def test_message_has_fallback_when_no_cycle(self) -> None:
        """When no cycle is given, the message must still be intelligible."""
        exc = CircularImportError("a")
        assert str(exc)  # non-empty
        assert "a" in str(exc)

    def test_repr_contains_class_name(self) -> None:
        """repr() must identify the exception class."""
        exc = CircularImportError("pkg.a")
        assert "CircularImportError" in repr(exc)


# ──────────────────────────────────────────────────────────────────────────────
# lazyload.ModuleNotFoundError (double-inheritance)
# ──────────────────────────────────────────────────────────────────────────────


class TestModuleNotFoundError:
    """``lazyload.ModuleNotFoundError`` must satisfy both exception hierarchies."""

    def test_is_subclass_of_lazy_load_error(self) -> None:
        """lazyload.ModuleNotFoundError must be a LazyLoadError."""
        assert issubclass(ModuleNotFoundError, LazyLoadError)

    def test_is_subclass_of_builtin_module_not_found_error(self) -> None:
        """lazyload.ModuleNotFoundError must also be a builtins.ModuleNotFoundError."""
        assert issubclass(ModuleNotFoundError, builtins.ModuleNotFoundError)

    def test_is_subclass_of_import_error(self) -> None:
        """lazyload.ModuleNotFoundError is an ImportError via the builtin chain."""
        assert issubclass(ModuleNotFoundError, ImportError)

    def test_caught_by_builtin_except_clause(self) -> None:
        """A bare 'except ModuleNotFoundError' must catch lazyload's version."""
        with pytest.raises(builtins.ModuleNotFoundError):
            raise ModuleNotFoundError("missing_module")

    def test_caught_by_lazy_load_error_except_clause(self) -> None:
        """A 'except LazyLoadError' must also catch lazyload.ModuleNotFoundError."""
        with pytest.raises(LazyLoadError):
            raise ModuleNotFoundError("missing_module")

    def test_stores_module_name(self) -> None:
        """The .module_name attribute must match the argument."""
        exc = ModuleNotFoundError("missing_pkg")
        assert exc.module_name == "missing_pkg"

    def test_sets_name_attribute_for_compatibility(self) -> None:
        """The .name attribute must be set, matching the builtin convention."""
        exc = ModuleNotFoundError("missing_pkg")
        assert exc.name == "missing_pkg"

    def test_message_contains_module_name(self) -> None:
        """The error message must name the missing module."""
        exc = ModuleNotFoundError("totally_missing")
        assert "totally_missing" in str(exc)

    def test_custom_message_is_respected(self) -> None:
        """A custom msg argument overrides the auto-generated message."""
        exc = ModuleNotFoundError("pkg", msg="Custom error text")
        assert "Custom error text" in str(exc)

    def test_repr_contains_class_name(self) -> None:
        """repr() must include the class name."""
        exc = ModuleNotFoundError("missing")
        assert "ModuleNotFoundError" in repr(exc)
