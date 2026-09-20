"""Tests for lazyload._compat — the deferred-proxy shim (Python 3.10–3.14).

The entire module is skipped on Python 3.15+ because importing ``_compat``
raises ``ImportError`` on that interpreter (its module-level guard fires).

Test strategy
-------------
* Use only stdlib modules as deferral targets — never third-party packages —
  so the suite runs in any environment without extra dependencies.
* ``fractions``, ``statistics``, and ``calendar`` are used as targets because
  they are not imported during pytest collection or by ``lazyload`` itself, so
  they are reliably absent from ``sys.modules`` at test start.
* The ``isolate_module`` fixture from ``conftest.py`` ensures each test gets a
  clean slate for its target module and restores ``sys.modules`` on teardown.
* The ``restore_import`` fixture (applied where needed) guarantees that
  ``builtins.__import__`` is always restored, even if a test fails mid-way.
"""

from __future__ import annotations

import builtins
from collections.abc import Callable
import sys
import types
from typing import Any
from unittest.mock import patch

import pytest

# Skip this entire module if we are on Python 3.15+.
if sys.version_info >= (3, 15):
    pytest.skip(
        allow_module_level=True,
        reason="lazyload._compat is only active on Python 3.10–3.14",
    )

# Safe to import after the skip guard.
from lazyload._compat import (
    _DeferredProxy,
    _lazy_compat,
    _lazy_module_compat,
    _LazyImportsCompat,
)

# ──────────────────────────────────────────────────────────────────────────────
# Sentinel module name used as a target in several tests.
# ``statistics`` is a good choice: not imported by lazyload or pytest itself.
# ──────────────────────────────────────────────────────────────────────────────
_TARGET = "statistics"
_TARGET_2 = "fractions"
_TARGET_3 = "calendar"


# ──────────────────────────────────────────────────────────────────────────────
# _DeferredProxy — construction
# ──────────────────────────────────────────────────────────────────────────────


class TestDeferredProxyConstruction:
    """Verify that a proxy is created correctly without loading the module."""

    def test_is_module_type(self, isolate_module: Any) -> None:
        """_DeferredProxy must be a subclass of types.ModuleType."""
        with isolate_module(_TARGET):
            proxy = _DeferredProxy(_TARGET)
            assert isinstance(proxy, types.ModuleType)

    def test_is_deferred_proxy(self, isolate_module: Any) -> None:
        """The proxy must be an instance of _DeferredProxy specifically."""
        with isolate_module(_TARGET):
            proxy = _DeferredProxy(_TARGET)
            assert isinstance(proxy, _DeferredProxy)

    def test_not_loaded_at_construction(self, isolate_module: Any) -> None:
        """Creating a proxy must not import the module — the loaded flag is False."""
        with isolate_module(_TARGET):
            proxy = _DeferredProxy(_TARGET)
            assert proxy.__dict__["__lazyload_loaded__"] is False

    def test_target_not_in_sys_modules_after_construction(
        self, isolate_module: Any
    ) -> None:
        """Constructing a _DeferredProxy must not register it in sys.modules."""
        with isolate_module(_TARGET):
            _DeferredProxy(_TARGET)
            # Construction alone must NOT add anything to sys.modules.
            assert _TARGET not in sys.modules

    def test_repr_shows_deferred_status(self, isolate_module: Any) -> None:
        """repr() before reification must contain the word 'deferred'."""
        with isolate_module(_TARGET):
            proxy = _DeferredProxy(_TARGET)
            assert "deferred" in repr(proxy).lower()

    def test_repr_contains_module_name(self, isolate_module: Any) -> None:
        """repr() must include the target module name."""
        with isolate_module(_TARGET):
            proxy = _DeferredProxy(_TARGET)
            assert _TARGET in repr(proxy)

    def test_relative_import_without_package_raises(self) -> None:
        """Creating a _DeferredProxy for '.utils' without package must raise ImportError."""
        with pytest.raises(ImportError, match="package"):
            _DeferredProxy(".utils")


# ──────────────────────────────────────────────────────────────────────────────
# _DeferredProxy — reification
# ──────────────────────────────────────────────────────────────────────────────


class TestDeferredProxyReification:
    """Accessing an attribute on a proxy must trigger a real import exactly once."""

    def test_attribute_access_triggers_load(self, isolate_module: Any) -> None:
        """Accessing any attribute on the proxy must cause the real module to load."""
        with isolate_module(_TARGET):
            proxy = _DeferredProxy(_TARGET)
            sys.modules[_TARGET] = proxy
            # `mean` is a function defined in the statistics module.
            _ = proxy.mean
            assert proxy.__dict__["__lazyload_loaded__"] is True

    def test_real_module_in_sys_modules_after_reification(
        self, isolate_module: Any
    ) -> None:
        """After reification, sys.modules must hold the real module, not the proxy."""
        with isolate_module(_TARGET):
            proxy = _DeferredProxy(_TARGET)
            sys.modules[_TARGET] = proxy
            _ = proxy.mean
            real = sys.modules[_TARGET]
            # The real module in sys.modules is not the proxy anymore.
            assert not isinstance(real, _DeferredProxy)
            assert isinstance(real, types.ModuleType)

    def test_attributes_are_correct_after_reification(
        self, isolate_module: Any
    ) -> None:
        """Attributes from the real module must be accessible through the proxy."""
        with isolate_module(_TARGET):
            proxy = _DeferredProxy(_TARGET)
            sys.modules[_TARGET] = proxy
            # statistics.mean exists and is callable.
            fn = proxy.mean
            assert callable(fn)
            assert fn([1, 2, 3]) == 2

    def test_subsequent_attribute_access_is_via_dict(self, isolate_module: Any) -> None:
        """After first reification, subsequent access must find attrs in __dict__."""
        with isolate_module(_TARGET):
            proxy = _DeferredProxy(_TARGET)
            sys.modules[_TARGET] = proxy
            _ = proxy.mean  # reify
            # After merging, 'mean' must be in the proxy's __dict__ directly.
            assert "mean" in proxy.__dict__

    def test_repr_shows_loaded_after_reification(self, isolate_module: Any) -> None:
        """repr() after reification must say '[loaded]', not '[deferred]'."""
        with isolate_module(_TARGET):
            proxy = _DeferredProxy(_TARGET)
            sys.modules[_TARGET] = proxy
            _ = proxy.mean
            r = repr(proxy)
            assert "[loaded]" in r.lower()
            assert "[deferred]" not in r.lower()

    def test_missing_attribute_raises_attribute_error(
        self, isolate_module: Any
    ) -> None:
        """Accessing a non-existent attribute after reification raises AttributeError."""
        with isolate_module(_TARGET):
            proxy = _DeferredProxy(_TARGET)
            sys.modules[_TARGET] = proxy
            with pytest.raises(AttributeError, match="no attribute"):
                _ = proxy.__totally_does_not_exist_xyz__

    def test_nonexistent_module_raises_on_attribute_access(
        self, isolate_module: Any
    ) -> None:
        """Reifying a proxy for a non-existent module must raise an error."""
        bad_name = "_lazyload_does_not_exist_at_all_xyz"
        with isolate_module(bad_name):
            proxy = _DeferredProxy(bad_name)
            sys.modules[bad_name] = proxy
            with pytest.raises((ModuleNotFoundError, ImportError)):
                _ = proxy.anything

    def test_proxy_restored_in_sys_modules_on_import_failure(
        self, isolate_module: Any
    ) -> None:
        """If reification fails, the proxy must be put back in sys.modules."""
        bad_name = "_lazyload_does_not_exist_at_all_xyz"
        with isolate_module(bad_name):
            proxy = _DeferredProxy(bad_name)
            sys.modules[bad_name] = proxy
            try:
                _ = proxy.anything
            except (ModuleNotFoundError, ImportError):
                pass
            # The proxy should be back in sys.modules after the failure.
            assert sys.modules.get(bad_name) is proxy

    def test_reify_is_idempotent(self, isolate_module: Any) -> None:
        """Calling _reify() multiple times must not re-import the module."""
        with isolate_module(_TARGET):
            proxy = _DeferredProxy(_TARGET)
            sys.modules[_TARGET] = proxy
            _ = proxy.mean  # first reification
            real_after_first = sys.modules[_TARGET]
            _ = proxy.mean  # second access
            real_after_second = sys.modules[_TARGET]
            # sys.modules must hold the same object both times.
            assert real_after_first is real_after_second


# ──────────────────────────────────────────────────────────────────────────────
# _lazy_compat() function
# ──────────────────────────────────────────────────────────────────────────────


class TestLazyCompat:
    """``_lazy_compat`` must create, reuse, or skip proxies correctly."""

    def test_returns_deferred_proxy_for_fresh_module(self, isolate_module: Any) -> None:
        """_lazy_compat on a module not in sys.modules must return a _DeferredProxy."""
        with isolate_module(_TARGET):
            result = _lazy_compat(_TARGET)
            assert isinstance(result, _DeferredProxy)

    def test_not_in_sys_modules_until_reification(self, isolate_module: Any) -> None:
        """_lazy_compat must not place the module in sys.modules until attribute access (reification)."""
        with isolate_module(_TARGET):
            proxy = _lazy_compat(_TARGET)
            assert _TARGET not in sys.modules
            _ = proxy.mean
            assert sys.modules.get(_TARGET) is not None
            assert not isinstance(sys.modules[_TARGET], _DeferredProxy)

    def test_returns_same_proxy_if_called_twice(self, isolate_module: Any) -> None:
        """Two calls to _lazy_compat for the same module return the same proxy."""
        with isolate_module(_TARGET):
            p1 = _lazy_compat(_TARGET)
            p2 = _lazy_compat(_TARGET)
            assert p1 is p2

    def test_returns_real_module_if_already_loaded(self) -> None:
        """If the module is already in sys.modules (real), return it directly."""
        # sys is always loaded — use it as the already-loaded target.
        result = _lazy_compat("sys")
        assert result is sys
        assert not isinstance(result, _DeferredProxy)

    def test_relative_import_without_package_raises(self) -> None:
        """Calling _lazy_compat('.utils') without package must raise ImportError."""
        with pytest.raises(ImportError, match="package"):
            _lazy_compat(".utils")


# ──────────────────────────────────────────────────────────────────────────────
# _LazyImportsCompat context manager
# ──────────────────────────────────────────────────────────────────────────────


class TestLazyImportsCompat:
    """The context manager must intercept bare imports and restore state on exit."""

    def test_patches_builtins_import_on_enter(self, restore_import: None) -> None:
        """Inside the with block, builtins.__import__ must be replaced."""
        original = builtins.__import__
        with _LazyImportsCompat() as ctx:
            assert builtins.__import__ is not original

    def test_restores_builtins_import_on_exit(self, restore_import: None) -> None:
        """After the with block, builtins.__import__ must be restored."""
        original = builtins.__import__
        with _LazyImportsCompat():
            pass
        assert builtins.__import__ is original

    def test_restores_on_exception(self, restore_import: None) -> None:
        """builtins.__import__ must be restored even if the block raises."""
        original = builtins.__import__
        try:
            with _LazyImportsCompat():
                raise ValueError("deliberate")
        except ValueError:
            pass
        assert builtins.__import__ is original

    def test_from_import_is_still_eager(
        self, isolate_module: Any, restore_import: None
    ) -> None:
        """'from X import Y' inside the block must still be eager."""
        with isolate_module(_TARGET):
            with _LazyImportsCompat():
                # This is a from-import; _LazyImportsCompat must not defer it.
                from statistics import mean
            # After the block, mean is the real function (not a proxy attr).
            assert callable(mean)
            assert mean([1, 2, 3]) == 2

    def test_exit_without_enter_raises_runtime_error(self) -> None:
        """Calling __exit__ without __enter__ must raise RuntimeError."""
        ctx = _LazyImportsCompat()
        with pytest.raises(RuntimeError, match="__enter__"):
            ctx.__exit__(None, None, None)

    def test_nesting_restores_outer_importer(self, restore_import: None) -> None:
        """Nested contexts must unwind the import stack correctly."""
        original = builtins.__import__
        with _LazyImportsCompat() as outer:
            importer_in_outer = builtins.__import__
            with _LazyImportsCompat() as inner:
                pass
            # After inner exits, we must be back to the outer lazy importer.
            assert builtins.__import__ is importer_in_outer
        # After outer exits, we must be back to the original.
        assert builtins.__import__ is original

    def test_context_manager_returns_self(self, restore_import: None) -> None:
        """__enter__ must return the context manager instance itself."""
        ctx = _LazyImportsCompat()
        result = ctx.__enter__()
        assert result is ctx
        ctx.__exit__(None, None, None)


# ──────────────────────────────────────────────────────────────────────────────
# _lazy_module_compat decorator
# ──────────────────────────────────────────────────────────────────────────────


class TestLazyModuleCompat:
    """The decorator must defer imports on the first call only."""

    def test_preserves_function_name(self) -> None:
        """@_lazy_module_compat must preserve __name__ via functools.wraps."""

        @_lazy_module_compat
        def my_function() -> None:
            pass

        assert my_function.__name__ == "my_function"

    def test_preserves_function_docstring(self) -> None:
        """@_lazy_module_compat must preserve __doc__ via functools.wraps."""

        @_lazy_module_compat
        def documented() -> None:
            """My docstring."""

        assert documented.__doc__ == "My docstring."

    def test_decorated_function_returns_value(
        self, restore_import: None, isolate_module: Any
    ) -> None:
        """The decorator must not swallow the return value."""

        @_lazy_module_compat
        def get_answer() -> int:
            return 42

        assert get_answer() == 42

    def test_first_call_enters_lazy_context(
        self, restore_import: None, isolate_module: Any
    ) -> None:
        """On the first call, builtins.__import__ must be temporarily patched."""
        import_states: list[bool] = []
        original = builtins.__import__

        @_lazy_module_compat
        def record_import_state() -> None:
            # While the function body is running, __import__ should be patched.
            import_states.append(builtins.__import__ is not original)

        record_import_state()
        assert import_states == [True], (
            "Expected __import__ to be patched during the first call"
        )

    def test_second_call_does_not_enter_lazy_context(
        self, restore_import: None
    ) -> None:
        """On the second and subsequent calls, __import__ must NOT be patched."""
        original = builtins.__import__
        import_states: list[bool] = []

        @_lazy_module_compat
        def record_import_state() -> None:
            import_states.append(builtins.__import__ is not original)

        record_import_state()  # first call  — patched
        record_import_state()  # second call — not patched
        assert import_states[0] is True, "First call must enter lazy context"
        assert import_states[1] is False, "Second call must NOT enter lazy context"

    def test_exception_in_first_call_restores_import(
        self, restore_import: None
    ) -> None:
        """If the first call raises, __import__ must still be restored."""
        original = builtins.__import__

        @_lazy_module_compat
        def bad_function() -> None:
            raise RuntimeError("boom")

        with pytest.raises(RuntimeError):
            bad_function()

        assert builtins.__import__ is original

    def test_handles_positional_and_keyword_args(self, restore_import: None) -> None:
        """The decorator must pass all args and kwargs through correctly."""

        @_lazy_module_compat
        def add(a: int, b: int = 0) -> int:
            return a + b

        assert add(3, b=4) == 7
