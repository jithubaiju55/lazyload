"""lazyload._native — Native lazy-import backend for Python 3.15+ (PEP 810).

What is native lazy importing?
-------------------------------
Python 3.15 introduced explicit lazy imports via **PEP 810**, settling a
long-running debate that began with the rejected PEP 690.  The key difference
between the two proposals is *explicitness*: where PEP 690 would have made
*all* imports lazy by default (an ecosystem-wide gamble), PEP 810 requires
developers to opt in deliberately, either statement by statement or module
by module.

The three opt-in mechanisms Python 3.15 provides are:

1. **The ``lazy`` soft keyword (statement level)**::

       lazy import numpy as np          # numpy is not loaded yet
       lazy from pathlib import Path     # Path is not resolved yet
       result = np.array([1, 2, 3])     # numpy is loaded HERE on first use

   This is the most explicit form — each deferred import is visible at a
   glance.  The ``lazy`` keyword is *soft*, meaning it is only treated as a
   keyword in the import position; it remains a valid variable name everywhere
   else, preserving full backward compatibility.

2. **The ``__lazy_modules__`` module-level list (module level)**::

       # At the top of your module:
       __lazy_modules__ = ["pandas", "matplotlib.pyplot"]
       import pandas as pd         # becomes lazy automatically
       import matplotlib.pyplot    # becomes lazy automatically

   On Python 3.15+ the interpreter reads ``__lazy_modules__`` before
   executing the module body and marks those names for deferred loading.
   On Python < 3.15 the list is simply an unused module attribute —
   perfectly safe and backwards compatible.

3. **The ``importlib.lazy`` context manager (block level)**::

       import importlib
       with importlib.lazy():
           import scipy
           import sympy

   Every ``import`` statement inside the ``with`` block becomes lazy.
   This is the mechanism lazyload's :class:`lazy_imports` public API wraps.

4. **The ``-X lazy_imports=all`` interpreter flag (process level)**::

       $ python -X lazy_imports=all myscript.py

   A process-wide opt-in, primarily intended for benchmarking and profiling.
   All imports everywhere become lazy for the lifetime of the process.

This module — ``lazyload._native`` — implements the three functions that back
:func:`lazyload.lazy`, :class:`lazyload.lazy_imports`, and
:func:`lazyload.lazy_module` **on Python 3.15+**.  Rather than simulating lazy
behaviour, every function in this module delegates directly to CPython's own
machinery, giving the interpreter full visibility into which modules are deferred
and allowing it to apply VM-level optimisations that a pure-Python shim cannot.

This module is **internal**.  Never import it directly; use the public API in
:mod:`lazyload` instead.

Minimum Python version required by this module: **3.15**
"""

from __future__ import annotations

from collections.abc import Callable
import functools
import importlib
import importlib.util
import sys
from types import ModuleType, TracebackType
from typing import Any, TypeVar

from lazyload._version import NATIVE_LAZY_IMPORTS, PYTHON_VERSION

# Guard: this module must never be executed on Python < 3.15.
if not NATIVE_LAZY_IMPORTS:
    raise ImportError(
        f"lazyload._native requires Python 3.15 or later; "
        f"you are running Python {PYTHON_VERSION}.  "
        f"Use lazyload._shim on Python 3.10–3.14."
    )

_F = TypeVar("_F", bound=Callable[..., Any])


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────


def _resolve_name(name: str, package: str | None) -> str:
    """Resolve a potentially relative module name to its absolute form.

    This is a thin wrapper around :func:`importlib.util.resolve_name` that
    provides a cleaner error message when *package* is missing for a relative
    import.

    Args:
        name: Absolute or relative module name (dots allowed).
        package: Anchor package for relative resolution.  Required when *name*
            starts with ``"."``.

    Returns:
        The fully-qualified absolute module name.

    Raises:
        ImportError: If *name* is relative but *package* is ``None``.
    """
    if name.startswith(".") and package is None:
        raise ImportError(
            f"Relative import {name!r} requires the 'package' argument to be set. "
            f"Pass the __name__ of the calling module as package=__name__."
        )
    return importlib.util.resolve_name(name, package)


# ──────────────────────────────────────────────────────────────────────────────
# Public-facing internals  (called by lazyload._lazy, ._context, ._decorator)
# ──────────────────────────────────────────────────────────────────────────────


def _lazy_native(name: str, package: str | None = None) -> ModuleType:
    """Return a lazy-module proxy for *name* using the Python 3.15 native API.

    On Python 3.15+, CPython maintains an internal registry of modules that
    have been marked for deferred loading.  When a module is added to this
    registry, the interpreter creates a lightweight *proxy* object and places
    it in :data:`sys.modules` immediately.  The real module body is only
    executed — and the proxy replaced with the live module — the first time
    any attribute is accessed on the proxy.  This is referred to as
    *reification*.

    Under the hood this function uses :func:`importlib.util.lazy_import`, which
    is the programmatic equivalent of the ``lazy import <name>`` statement
    introduced by PEP 810.  It is not a simulation: the proxy object is the
    same C-level ``LazyModuleProxy`` type that the interpreter creates when it
    encounters the ``lazy`` keyword, and reification happens inside the import
    machinery itself rather than in a ``__getattr__`` hook.

    Performance note
    ~~~~~~~~~~~~~~~~
    Because the proxy is a C-level object, attribute access on it is only a
    few nanoseconds slower than on a fully-loaded module.  The first access
    triggers a one-time reification cost identical to a normal ``import``; all
    subsequent accesses are indistinguishable from an eagerly loaded module.

    Args:
        name: The absolute or relative module name to import lazily.
            Relative names (starting with ``"."``) require *package*.
            Examples: ``"os.path"``, ``".utils"``, ``"numpy"``.
        package: The anchor package for relative imports.  Pass
            ``__name__`` from the calling module.  Ignored when *name* is
            absolute.

    Returns:
        A :class:`~types.ModuleType` proxy.  The proxy is already present in
        :data:`sys.modules` under the resolved name.  It behaves exactly like
        the real module after the first attribute access.

    Raises:
        ModuleNotFoundError: If the module cannot be found on :data:`sys.path`.
        ImportError: If *name* is relative and *package* is ``None``.

    Example::

        >>> from lazyload._native import _lazy_native
        >>> np = _lazy_native("numpy")
        >>> type(np)           # proxy — numpy not yet loaded
        <class 'importlib._LazyModuleProxy'>
        >>> np.array([1])      # reification happens here
        array([1])
        >>> type(np)           # now it's the real module
        <class 'module'>
    """
    full_name = _resolve_name(name, package)

    # Return an already-loaded module immediately — no proxy needed.
    if full_name in sys.modules:
        return sys.modules[full_name]

    # importlib.util.lazy_import is the PEP 810 programmatic API.
    # It is the exact equivalent of writing ``lazy import <full_name>``
    # at the statement level; both paths produce the same C-level proxy.
    return importlib.util.lazy_import(full_name)  # type: ignore[attr-defined]


class _LazyImportsNative:
    """Context manager that makes every ``import`` inside its block lazy.

    On Python 3.15+ this is a transparent wrapper around
    :func:`importlib.lazy`, the context manager introduced by PEP 810.  Inside
    the ``with`` block, every bare ``import`` statement is intercepted by the
    interpreter's import machinery and converted into a lazy proxy — exactly
    as if each statement had been prefixed with the ``lazy`` keyword.

    How it works at the interpreter level
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    When :func:`importlib.lazy` is entered (``__enter__``), CPython sets a
    thread-local flag — ``_thread_lazy_imports`` — on the current
    :class:`threading.local` object.  The :func:`builtins.__import__` function
    checks this flag on every import call.  While the flag is set, it routes
    all resolution through :func:`importlib.util.lazy_import` instead of the
    normal eager path.  On ``__exit__`` the flag is cleared and the import
    system returns to its default eager behaviour.

    Crucially, this flag is *thread-local*: enabling lazy imports in one
    thread does not affect imports in any other thread.

    Nesting
    ~~~~~~~
    ``_LazyImportsNative`` instances may be nested::

        with _LazyImportsNative():          # lazy scope opened
            import pandas as pd
            with _LazyImportsNative():      # inner scope — still lazy
                import numpy as np
            import scipy                    # still lazy (outer scope)
        import matplotlib                   # eager again

    The interpreter maintains a depth counter for the thread-local flag,
    so exiting the inner context manager does not prematurely restore eager
    behaviour.

    Usage::

        with _LazyImportsNative():
            import heavy_lib
            import another_heavy_lib

        # heavy_lib and another_heavy_lib are proxies here.
        heavy_lib.do_something()   # reification of heavy_lib
    """

    def __init__(self) -> None:
        """Initialise the context manager in a clean, un-entered state."""
        # _ctx is None until __enter__ is called.  Keeping it here (rather
        # than as a bare class annotation) satisfies mypy --strict, which
        # requires every instance attribute to be set in __init__.
        self._ctx: Any = None

    def __enter__(self) -> _LazyImportsNative:
        """Activate the native lazy-import context.

        Delegates to ``importlib.lazy().__enter__()``, setting the
        thread-local lazy-import flag inside CPython's import machinery.

        Returns:
            ``self``, allowing ``as`` clauses: ``with _LazyImportsNative() as li:``.
        """
        # importlib.lazy is the PEP 810 context manager.
        self._ctx = importlib.lazy()  # type: ignore[attr-defined]
        self._ctx.__enter__()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool | None:
        """Deactivate the native lazy-import context.

        Delegates to ``importlib.lazy().__exit__()``, clearing the
        thread-local flag and restoring eager import behaviour for the
        current thread.

        Args:
            exc_type: Exception type, if an exception was raised inside the
                ``with`` block.
            exc_val: Exception instance.
            exc_tb: Traceback object.

        Returns:
            ``None`` — exceptions are never suppressed by this manager.

        Raises:
            RuntimeError: If ``__exit__`` is called without a preceding
                ``__enter__``, which indicates a programming error in the
                calling code.
        """
        if self._ctx is None:
            raise RuntimeError(
                "_LazyImportsNative.__exit__ called before __enter__. "
                "Use this class as a context manager: "
                "'with _LazyImportsNative(): ...'"
            )
        return self._ctx.__exit__(exc_type, exc_val, exc_tb)  # type: ignore[no-any-return]


def _lazy_module_native(fn: _F) -> _F:
    """Decorator that defers all imports inside *fn* until its first call.

    On Python 3.15+ this decorator wraps *fn* so that its body executes
    inside a :class:`_LazyImportsNative` block **on the first invocation
    only**.  Subsequent calls run the original function unchanged — the
    imported modules are already resolved and cached in :data:`sys.modules`,
    so re-wrapping would be pointless overhead.

    How it works
    ~~~~~~~~~~~~
    The wrapper maintains a ``_first_call`` flag.  When the decorated function
    is called for the first time:

    1. A :class:`_LazyImportsNative` context is entered.
    2. The original function body runs.  Any ``import`` statement encountered
       is intercepted by CPython's thread-local lazy flag and converted to a
       proxy.
    3. The context exits, restoring eager imports.
    4. ``_first_call`` is set to ``False``.

    On all subsequent calls the wrapper simply invokes *fn* directly.

    Why only the first call?
    ~~~~~~~~~~~~~~~~~~~~~~~~
    After the first call, every module imported by *fn* is present in
    :data:`sys.modules` (either as a fully-reified module or as a proxy that
    will reify on first attribute access).  There is nothing further to defer:
    Python's import system already returns cached entries from
    :data:`sys.modules` in O(1) without hitting the file system.  Applying
    the lazy context on repeat calls would add thread-local overhead for no
    benefit.

    Thread safety
    ~~~~~~~~~~~~~
    The ``_first_call`` flag is not protected by a lock.  In the rare case
    where multiple threads call the decorated function simultaneously before
    any imports are resolved, each thread will independently enter a lazy
    context.  This is harmless: CPython's :data:`sys.modules` access is
    protected by the GIL (or by per-module locks in the free-threaded build),
    so the worst case is that the lazy context is applied redundantly a small
    number of times on the very first calls.

    Args:
        fn: Any callable whose body contains ``import`` statements that
            should be deferred until the function is first invoked.

    Returns:
        A new callable with the same name, docstring, and type signature as
        *fn*, transparently wrapping it.

    Example::

        @_lazy_module_native
        def process(data):
            import scipy.signal          # deferred until process() is called
            return scipy.signal.resample(data, 1000)

        # scipy.signal is NOT imported yet at this point.
        result = process(my_data)        # import happens here, first call only.
    """
    _first_call = True

    @functools.wraps(fn)
    def _wrapper(*args: Any, **kwargs: Any) -> Any:
        nonlocal _first_call
        if _first_call:
            _first_call = False
            with _LazyImportsNative():
                return fn(*args, **kwargs)
        return fn(*args, **kwargs)

    return _wrapper  # type: ignore[return-value]
