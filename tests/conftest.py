"""Shared pytest fixtures for the lazyload test suite.

Fixture design principles
-------------------------
1. Every test that touches ``sys.modules`` must leave it in the same state it
   found it.  The ``isolate_module`` fixture ensures this.
2. Version-detection constants in ``lazyload._version`` are computed once at
   import time and cannot be changed by patching ``sys.version_info`` after
   the fact.  The ``simulate_py310`` and ``simulate_py315`` fixtures patch the
   *already-computed* module-level names directly using ``monkeypatch``.
3. Fixtures that touch global state (``builtins.__import__``, ``sys.modules``)
   restore it unconditionally, even if the test raises.
"""

from __future__ import annotations

import builtins
from contextlib import contextmanager
import sys
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Callable, Generator
    from contextlib import AbstractContextManager
    from types import ModuleType


# ──────────────────────────────────────────────────────────────────────────────
# sys.modules isolation
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def isolate_module() -> Callable[..., AbstractContextManager[None]]:
    """Return a context manager that temporarily removes named modules from
    ``sys.modules`` and restores the original state on exit.

    Usage inside a test::

        def test_something(isolate_module):
            with isolate_module("statistics", "fractions"):
                # Both modules are absent from sys.modules here.
                ...
            # Both modules (or their absence) are restored here.
    """
    @contextmanager
    def _ctx(*names: str) -> Generator[None, None, None]:
        saved: dict[str, ModuleType | None] = {
            name: sys.modules.pop(name, None) for name in names
        }
        try:
            yield
        finally:
            for name, mod in saved.items():
                if mod is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = mod

    return _ctx


# ──────────────────────────────────────────────────────────────────────────────
# builtins.__import__ isolation
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=False)
def restore_import() -> Generator[None, None, None]:
    """Guarantee that ``builtins.__import__`` is restored after each test.

    Use this fixture in any test that exercises ``_LazyImportsCompat`` or any
    other code that patches the import function, so that a failing test cannot
    leave the import system in a broken state for subsequent tests.
    """
    original = builtins.__import__
    yield
    builtins.__import__ = original


# ──────────────────────────────────────────────────────────────────────────────
# Version-simulation fixtures
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def simulate_py310(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch ``lazyload._version`` to appear as if running on Python 3.10.0.

    Because ``_version.py`` evaluates its constants at *import time*, simply
    patching ``sys.version_info`` has no effect.  This fixture patches the
    already-computed module-level constants directly.

    After this fixture is applied:
    * ``lazyload._version.NATIVE_LAZY_IMPORTS`` is ``False``
    * ``lazyload._version.SHIM_REQUIRED``       is ``True``
    * ``lazyload._version.PYTHON_VERSION``       is ``"3.10.0"``
    * ``lazyload._version.MODE``                 is ``"shim"``
    """
    import lazyload._version as _v

    monkeypatch.setattr(_v, "NATIVE_LAZY_IMPORTS", False)
    monkeypatch.setattr(_v, "SHIM_REQUIRED", True)
    monkeypatch.setattr(_v, "PYTHON_VERSION", "3.10.0")
    monkeypatch.setattr(_v, "MODE", "shim")


@pytest.fixture
def simulate_py315(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch ``lazyload._version`` to appear as if running on Python 3.15.0.

    After this fixture is applied:
    * ``lazyload._version.NATIVE_LAZY_IMPORTS`` is ``True``
    * ``lazyload._version.SHIM_REQUIRED``       is ``False``
    * ``lazyload._version.PYTHON_VERSION``       is ``"3.15.0"``
    * ``lazyload._version.MODE``                 is ``"native"``
    """
    import lazyload._version as _v

    monkeypatch.setattr(_v, "NATIVE_LAZY_IMPORTS", True)
    monkeypatch.setattr(_v, "SHIM_REQUIRED", False)
    monkeypatch.setattr(_v, "PYTHON_VERSION", "3.15.0")
    monkeypatch.setattr(_v, "MODE", "native")


# ──────────────────────────────────────────────────────────────────────────────
# Backend availability markers
# ──────────────────────────────────────────────────────────────────────────────

#: Marker applied to tests that require the compat shim (Python 3.10-3.14).
requires_compat = pytest.mark.skipif(
    sys.version_info >= (3, 15),
    reason="Compat shim is only active on Python 3.10-3.14",
)

#: Marker applied to tests that require the native backend (Python 3.15+).
requires_native = pytest.mark.skipif(
    sys.version_info < (3, 15),
    reason="Native backend is only active on Python 3.15+",
)
