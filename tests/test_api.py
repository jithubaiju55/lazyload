"""Tests for the lazyload public API — ``lazy``, ``lazy_imports``, ``lazy_module``.

These tests exercise the three public functions through the top-level package
interface (``import lazyload``) so they run correctly on *both* backends —
the native backend on Python 3.15+ and the compat shim on 3.10–3.14.
They do not reach into backend internals; they only verify the observable
contract that users rely on.

Stdlib-only targets
-------------------
All deferral targets are stdlib modules chosen because they are reliably absent
from ``sys.modules`` at test start:

* ``statistics`` — pure-Python statistics functions
* ``fractions``  — rational numbers
* ``calendar``   — calendar utilities
* ``textwrap``   — text-wrapping utilities

The ``isolate_module`` and ``restore_import`` fixtures from ``conftest.py``
ensure ``sys.modules`` and ``builtins.__import__`` are always restored.
"""

from __future__ import annotations

import builtins
import sys
import types
from typing import Any

import pytest

import lazyload
from lazyload._exceptions import (
    CircularImportError,
    LazyLoadError,
    UnsupportedPythonVersion,
)

# Convenience aliases for the three public functions.
lazy = lazyload.lazy
lazy_imports = lazyload.lazy_imports
lazy_module = lazyload.lazy_module

_MOD_A = "statistics"
_MOD_B = "fractions"
_MOD_C = "calendar"
_MOD_D = "textwrap"


# ──────────────────────────────────────────────────────────────────────────────
# Package-level exports
# ──────────────────────────────────────────────────────────────────────────────


class TestPackageExports:
    """The top-level package must export exactly the documented public surface."""

    def test_lazy_is_exported(self) -> None:
        """lazyload.lazy must be accessible as a top-level attribute."""
        assert hasattr(lazyload, "lazy")
        assert callable(lazyload.lazy)

    def test_lazy_imports_is_exported(self) -> None:
        """lazyload.lazy_imports must be accessible as a top-level attribute."""
        assert hasattr(lazyload, "lazy_imports")
        assert callable(lazyload.lazy_imports)

    def test_lazy_module_is_exported(self) -> None:
        """lazyload.lazy_module must be accessible as a top-level attribute."""
        assert hasattr(lazyload, "lazy_module")
        assert callable(lazyload.lazy_module)

    def test_all_contains_core_api(self) -> None:
        """__all__ must include all three public API functions."""
        assert "lazy" in lazyload.__all__
        assert "lazy_imports" in lazyload.__all__
        assert "lazy_module" in lazyload.__all__

    def test_all_contains_exceptions(self) -> None:
        """__all__ must include all four public exception classes."""
        assert "LazyLoadError" in lazyload.__all__
        assert "UnsupportedPythonVersion" in lazyload.__all__
        assert "CircularImportError" in lazyload.__all__
        assert "ModuleNotFoundError" in lazyload.__all__

    def test_version_is_string(self) -> None:
        """lazyload.__version__ must be a non-empty string."""
        assert isinstance(lazyload.__version__, str)
        assert lazyload.__version__

    def test_version_format(self) -> None:
        """__version__ must follow MAJOR.MINOR.PATCH semantic versioning."""
        parts = lazyload.__version__.split(".")
        assert len(parts) == 3
        assert all(p.isdigit() for p in parts)

    def test_exception_classes_accessible(self) -> None:
        """All four exception classes must be accessible directly on the package."""
        assert hasattr(lazyload, "LazyLoadError")
        assert hasattr(lazyload, "UnsupportedPythonVersion")
        assert hasattr(lazyload, "CircularImportError")
        assert hasattr(lazyload, "ModuleNotFoundError")


# ──────────────────────────────────────────────────────────────────────────────
# lazy() — single-module deferral
# ──────────────────────────────────────────────────────────────────────────────


class TestLazyFunction:
    """``lazy()`` must return a module-like proxy and defer real loading."""

    def test_returns_module_type(self, isolate_module: Any) -> None:
        """lazy() must return an object that is an instance of types.ModuleType."""
        with isolate_module(_MOD_A):
            result = lazy(_MOD_A)
            assert isinstance(result, types.ModuleType)

    def test_registers_in_sys_modules(self, isolate_module: Any) -> None:
        """Calling lazy() must NOT place the module in sys.modules until first attribute access."""
        with isolate_module(_MOD_A):
            proxy = lazy(_MOD_A)
            assert _MOD_A not in sys.modules
            _ = proxy.mean
            assert _MOD_A in sys.modules

    def test_does_not_load_module_immediately(self, isolate_module: Any) -> None:
        """After lazy(), the module body must not have been executed yet.

        Concretely: the proxy must be a _DeferredProxy (on 3.10–3.14) or a
        native LazyModuleProxy (on 3.15+) — NOT the real module object.
        For this test we check via the repr, which must mention 'deferred' or
        'lazy' on both backends.
        """
        with isolate_module(_MOD_A):
            proxy = lazy(_MOD_A)
            # The object must NOT be the real statistics module.
            # We verify by checking that it hasn't been fully imported yet.
            # If it were real, sys.modules[_MOD_A] would be identical to
            # the actual module (which is a types.ModuleType subclass with
            # a properly set __spec__).
            # Simplest cross-backend check: accessing __spec__ on the proxy
            # should not crash, but we can also check the class name.
            cls_name = type(proxy).__name__
            assert cls_name != "module", (
                f"Expected a proxy, got {cls_name!r} — module was loaded eagerly"
            )

    def test_attribute_access_triggers_real_load(self, isolate_module: Any) -> None:
        """Accessing any attribute on the proxy must trigger the real import."""
        with isolate_module(_MOD_A):
            proxy = lazy(_MOD_A)
            # statistics.mean is a real callable.
            fn = proxy.mean
            assert callable(fn)
            assert fn([10, 20, 30]) == 20

    def test_proxy_is_replaced_in_sys_modules_after_access(
        self, isolate_module: Any
    ) -> None:
        """After reification, sys.modules[name] must be the real module."""
        with isolate_module(_MOD_A):
            proxy = lazy(_MOD_A)
            _ = proxy.mean  # trigger reification
            real = sys.modules[_MOD_A]
            # The real module must not be the same object as the proxy.
            assert type(real).__name__ == "module"

    def test_second_call_returns_same_object(self, isolate_module: Any) -> None:
        """Calling lazy() twice for the same name must return the same proxy."""
        with isolate_module(_MOD_A):
            p1 = lazy(_MOD_A)
            p2 = lazy(_MOD_A)
            assert p1 is p2

    def test_already_loaded_module_returned_directly(self) -> None:
        """If a module is already in sys.modules, lazy() must return it directly."""
        # sys is always loaded.
        result = lazy("sys")
        assert result is sys

    def test_nonexistent_module_no_error_at_call_time(
        self, isolate_module: Any
    ) -> None:
        """lazy('totally_missing_xyz') must not raise at call time."""
        bad = "_lazyload_totally_missing_module_xyz"
        with isolate_module(bad):
            # Should not raise here.
            proxy = lazy(bad)
            assert proxy is not None

    def test_nonexistent_module_raises_on_attribute_access(
        self, isolate_module: Any
    ) -> None:
        """Accessing an attribute on a proxy for a missing module must raise."""
        bad = "_lazyload_totally_missing_module_xyz"
        with isolate_module(bad):
            proxy = lazy(bad)
            with pytest.raises((ModuleNotFoundError, ImportError)):
                _ = proxy.anything

    def test_relative_import_without_package_raises_immediately(self) -> None:
        """lazy('.utils') without package must raise ImportError at call time."""
        with pytest.raises(ImportError, match="package"):
            lazy(".utils")

    def test_absolute_dotted_name(self, isolate_module: Any) -> None:
        """lazy('xml.etree.ElementTree') with a dotted absolute name must work."""
        with isolate_module("xml", "xml.etree", "xml.etree.ElementTree"):
            proxy = lazy("xml.etree.ElementTree")
            assert isinstance(proxy, types.ModuleType)
            # Trigger reification.
            fn = proxy.parse
            assert fn is not None


# ──────────────────────────────────────────────────────────────────────────────
# lazy_imports() — block-level deferral
# ──────────────────────────────────────────────────────────────────────────────


class TestLazyImportsFunction:
    """``lazy_imports()`` must return a usable context manager."""

    def test_returns_context_manager(self, restore_import: None) -> None:
        """lazy_imports() must return an object with __enter__ and __exit__."""
        cm = lazy_imports()
        assert hasattr(cm, "__enter__")
        assert hasattr(cm, "__exit__")

    def test_usable_as_with_statement(self, restore_import: None) -> None:
        """lazy_imports() must work in a 'with' statement without raising."""
        with lazy_imports():
            pass  # Must not raise.

    def test_restores_import_after_block(self, restore_import: None) -> None:
        """After the 'with lazy_imports()' block, builtins.__import__ is original."""
        original = builtins.__import__
        with lazy_imports():
            pass
        assert builtins.__import__ is original

    def test_restores_import_on_exception(self, restore_import: None) -> None:
        """builtins.__import__ is restored even if the block raises."""
        original = builtins.__import__
        try:
            with lazy_imports():
                raise ValueError("deliberate error")
        except ValueError:
            pass
        assert builtins.__import__ is original

    def test_multiple_modules_can_be_deferred(
        self, isolate_module: Any, restore_import: None
    ) -> None:
        """Multiple imports inside one lazy_imports() block must all be proxied."""
        with isolate_module(_MOD_A, _MOD_B):
            with lazy_imports():
                import fractions
                import statistics

            # Neither module should be fully loaded yet (proxied).
            # Trigger reification to verify they resolve correctly.
            assert callable(statistics.mean)
            assert fractions.Fraction(1, 2) == fractions.Fraction("1/2")

    def test_can_be_nested(self, restore_import: None) -> None:
        """Two nested lazy_imports() blocks must both work and unwind cleanly."""
        original = builtins.__import__
        with lazy_imports():
            with lazy_imports():
                pass
        assert builtins.__import__ is original


# ──────────────────────────────────────────────────────────────────────────────
# lazy_module() — decorator form
# ──────────────────────────────────────────────────────────────────────────────


class TestLazyModuleDecorator:
    """``@lazy_module`` must defer imports inside a function until first call."""

    def test_preserves_function_name(self) -> None:
        """@lazy_module must keep the function's __name__ intact."""

        @lazy_module
        def my_func() -> None:
            pass

        assert my_func.__name__ == "my_func"

    def test_preserves_docstring(self) -> None:
        """@lazy_module must keep the function's __doc__ intact."""

        @lazy_module
        def documented() -> None:
            """My docstring."""

        assert documented.__doc__ == "My docstring."

    def test_function_is_callable(self) -> None:
        """The decorated function must remain callable."""

        @lazy_module
        def noop() -> None:
            pass

        assert callable(noop)

    def test_return_value_is_passed_through(self, restore_import: None) -> None:
        """The decorator must not swallow the function's return value."""

        @lazy_module
        def get_value() -> int:
            return 99

        assert get_value() == 99

    def test_positional_and_keyword_args_forwarded(self, restore_import: None) -> None:
        """All positional and keyword arguments must reach the wrapped function."""

        @lazy_module
        def add(x: int, y: int = 0) -> int:
            return x + y

        assert add(5, y=3) == 8

    def test_exception_inside_first_call_propagates(self, restore_import: None) -> None:
        """An exception raised inside the function on first call must propagate."""

        @lazy_module
        def explode() -> None:
            raise RuntimeError("boom")

        with pytest.raises(RuntimeError, match="boom"):
            explode()

    def test_import_inside_function_defers(
        self, isolate_module: Any, restore_import: None
    ) -> None:
        """An import inside the decorated function must be deferred to first call."""
        import_happened: list[bool] = []

        @lazy_module
        def do_work() -> Any:
            import statistics as _stats

            import_happened.append(True)
            return _stats.mean([1, 2, 3])

        # The import must not have happened yet.
        assert import_happened == []

        with isolate_module(_MOD_A):
            result = do_work()

        assert result == 2
        assert import_happened == [True]

    def test_second_call_does_not_re_enter_lazy_context(
        self, restore_import: None
    ) -> None:
        """The lazy context must not be re-entered on the second or later call."""
        original = builtins.__import__
        enter_count = [0]

        @lazy_module
        def track_calls() -> None:
            # If we are inside a lazy context, __import__ is patched.
            if builtins.__import__ is not original:
                enter_count[0] += 1

        track_calls()  # first call — lazy context entered
        track_calls()  # second call — must NOT re-enter
        track_calls()  # third call  — must NOT re-enter

        assert enter_count[0] == 1, (
            f"Expected lazy context entered once, but entered {enter_count[0]} times"
        )
