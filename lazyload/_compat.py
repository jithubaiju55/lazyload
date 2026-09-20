"""lazyload._compat — Deferred-proxy shim for Python 3.10–3.14.

Why a shim is needed
--------------------
Python 3.10 through 3.14 have no built-in lazy import syntax.  The ``lazy``
keyword and the ``importlib.lazy()`` context manager introduced by PEP 810 do
not exist on these versions.  To provide the same developer experience across
all supported interpreters, lazyload implements its own approximation using
two standard Python mechanisms that have been stable since Python 3.1:

1. ``sys.modules`` — the global module cache that the import system consults
   before touching the file system.
2. ``builtins.__import__`` — the function Python calls for every ``import``
   statement; it can be temporarily replaced to alter import behaviour.

The deferred-proxy pattern
--------------------------
The core building block is :class:`_DeferredProxy`, a subclass of
:class:`types.ModuleType` that registers itself in :data:`sys.modules` under
the target module's name *without executing the module's code*.  Because it
is already in ``sys.modules``, any subsequent ``import X`` statement that
Python's import machinery encounters will find the proxy immediately and
return it, never reading the file system.

The proxy has a single responsibility: the moment any *attribute* on it is
accessed, it performs the real import (called *reification*) by:

1. Temporarily removing itself from ``sys.modules`` so the import system
   runs normally.
2. Calling :func:`importlib.import_module` to load the real module.
3. Registering the real module in ``sys.modules`` under the module's name,
   replacing the proxy.
4. Merging the real module's ``__dict__`` into its own, so that **all
   subsequent attribute accesses are O(1) direct dict lookups** — the
   ``__getattr__`` hook is never called again for those names.

Is it a safe approximation?
----------------------------
For the overwhelming majority of use cases, yes.  There are a handful of
edge cases to be aware of:

* **Identity checks** — ``proxy is sys.modules["numpy"]`` is ``True`` before
  reification but ``False`` after, because ``sys.modules["numpy"]`` is
  replaced with the real module while the local variable still holds the
  proxy.  This is the same behaviour as ``importlib.util.LazyLoader``.

* **``isinstance`` checks** — :class:`_DeferredProxy` subclasses
  :class:`~types.ModuleType`, so ``isinstance(proxy, types.ModuleType)``
  returns ``True`` even before reification.

* **Pickling** — the proxy delegates :meth:`__reduce__` to the real module
  (triggering reification first), so ``pickle.dumps(proxy)`` works correctly.

* **``from X import Y``** — the context manager cannot defer these
  efficiently without loading the parent module anyway, so they are left
  eager.  Only bare ``import X`` statements are intercepted.

This module is **internal**.  Use the public API in :mod:`lazyload` instead.

Minimum Python version required: **3.10**.  Maximum: **3.14** (3.15+ uses
:mod:`lazyload._native`).
"""

from __future__ import annotations

import builtins
from collections.abc import Callable
import functools
import importlib
import importlib.util
import sys
from types import ModuleType, TracebackType
from typing import Any, TypeVar

from lazyload._version import PYTHON_VERSION, SHIM_REQUIRED

# Guard: this module must never be loaded on Python 3.15+.
if not SHIM_REQUIRED:
    raise ImportError(
        f"lazyload._compat is the compatibility shim for Python 3.10–3.14; "
        f"you are running Python {PYTHON_VERSION}.  "
        f"lazyload._native should have been selected instead."
    )

_F = TypeVar("_F", bound=Callable[..., Any])

# Prefix for internal sentinel keys stored inside the proxy's __dict__.
# Chosen to be unlikely to collide with any real module attribute.
_SENTINEL_PREFIX = "__lazyload_"

# Internal registry mapping module names to deferred proxies.
# Keeps sys.modules clean (not in sys.modules until reification).
_PROXY_CACHE: dict[str, _DeferredProxy] = {}


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _resolve_full_name(name: str, package: str | None) -> str:
    """Return the absolute module name for *name*, resolving relative dots.

    Args:
        name: Absolute or relative module name.  Relative names start with
            one or more dots (e.g. ``".utils"`` or ``"..core"``).
        package: Anchor package required for relative resolution.  Must be
            set when *name* starts with ``"."``.

    Returns:
        The fully-qualified absolute module name (e.g. ``"mypackage.utils"``).

    Raises:
        ImportError: If *name* is relative and *package* is ``None``.
    """
    if name.startswith(".") and package is None:
        raise ImportError(
            f"Relative import {name!r} requires the 'package' argument.  "
            f"Pass __name__ from the calling module: "
            f"lazy({name!r}, package=__name__)."
        )
    return importlib.util.resolve_name(name, package)


# ──────────────────────────────────────────────────────────────────────────────
# _DeferredProxy
# ──────────────────────────────────────────────────────────────────────────────

class _DeferredProxy(ModuleType):
    """A module-shaped proxy that defers the real import until first use.

    :class:`_DeferredProxy` is a :class:`~types.ModuleType` subclass that
    sits in :data:`sys.modules` under the target module's fully-qualified name.
    It carries no module code — only enough metadata to perform the real
    import on demand.

    Reification
    ~~~~~~~~~~~
    The first time any attribute is accessed on the proxy, :meth:`_reify` is
    called.  It:

    1. Pops the proxy from :data:`sys.modules` (so the import system won't
       return the proxy recursively).
    2. Calls :func:`importlib.import_module` to load the real module.
    3. Writes the real module back into :data:`sys.modules`.
    4. Merges the real module's ``__dict__`` into the proxy's own
       ``__dict__``, making all subsequent attribute accesses direct O(1)
       lookups — ``__getattr__`` is not invoked for those names again.

    Internal state
    ~~~~~~~~~~~~~~
    The proxy stores its own bookkeeping keys directly in ``__dict__`` using
    the ``__lazyload_*`` prefix, which is unlikely to collide with any real
    module attribute.  All access to these keys bypasses ``__getattr__``
    because Python's attribute lookup chain checks ``__dict__`` before
    calling ``__getattr__``.

    Thread safety
    ~~~~~~~~~~~~~
    Reification is not protected by an explicit lock.  In a multi-threaded
    programme where two threads simultaneously access the same proxy for the
    first time, both may race into :meth:`_reify`.  This is safe because:

    * :data:`sys.modules` mutations are protected by the GIL on CPython
      (or by per-import locks in free-threaded builds).
    * :func:`importlib.import_module` itself is re-entrant and idempotent:
      if the module is already being imported by another thread,
      :func:`importlib.import_module` waits on the per-module lock.
    * The worst case is that the ``__dict__`` merge happens twice, which is
      harmless.

    Example::

        proxy = _DeferredProxy("numpy")
        sys.modules["numpy"] = proxy
        repr(proxy)           # <_DeferredProxy 'numpy' [deferred]>
        _ = proxy.array       # triggers reification
        repr(proxy)           # <_DeferredProxy 'numpy' [loaded]>
        proxy.array           # O(1) — found directly in __dict__
    """

    def __init__(self, name: str, package: str | None = None) -> None:
        """Create a deferred proxy for the module *name*.

        Args:
            name: The module name to defer.  May be absolute (``"numpy"``) or
                relative (``".utils"``).  Relative names require *package*.
            package: Anchor package for relative name resolution.
        """
        full_name = _resolve_full_name(name, package)

        # Initialise the ModuleType base with the *original* (possibly relative)
        # name so that __name__ reflects what the caller asked for.
        super().__init__(full_name)

        # Store sentinel keys directly in __dict__ — bypasses __setattr__,
        # never triggers __getattr__, and survives the __dict__ merge below.
        self.__dict__[f"{_SENTINEL_PREFIX}full_name__"] = full_name
        self.__dict__[f"{_SENTINEL_PREFIX}loaded__"] = False

    # ------------------------------------------------------------------
    # Core reification logic
    # ------------------------------------------------------------------

    def _reify(self) -> ModuleType:
        """Load the real module and merge its attributes into this proxy.

        After :meth:`_reify` returns, every attribute that exists on the real
        module is present in the proxy's ``__dict__``.  Subsequent accesses
        to those names resolve via direct dict lookup and never reach
        ``__getattr__`` again.

        Returns:
            The *real* :class:`~types.ModuleType` object now registered in
            :data:`sys.modules`.

        Raises:
            ModuleNotFoundError: If the module cannot be located.
            ImportError: If the module exists but fails to load.
        """
        # Fast-path: already reified — real module attributes are in __dict__.
        if self.__dict__[f"{_SENTINEL_PREFIX}loaded__"]:
            # Return the real module from sys.modules; it may differ from self
            # because sys.modules was updated during reification.
            full_name: str = self.__dict__[f"{_SENTINEL_PREFIX}full_name__"]
            return sys.modules.get(full_name, self)

        full_name = self.__dict__[f"{_SENTINEL_PREFIX}full_name__"]

        # Evict from internal proxy cache
        _PROXY_CACHE.pop(full_name, None)

        # Remove the proxy from sys.modules BEFORE calling import_module.
        sys.modules.pop(full_name, None)

        try:
            real: ModuleType = importlib.import_module(full_name)
        except Exception:
            # Restore the proxy so that a later retry can attempt loading
            # again rather than raising a confusing KeyError from sys.modules.
            sys.modules[full_name] = self
            raise

        # Register the real module.  importlib.import_module already does
        # this, but we do it explicitly in case a custom loader skipped it.
        sys.modules[full_name] = real

        # Merge the real module's public namespace into the proxy's __dict__.
        # This makes all subsequent attribute accesses O(1) dict lookups.
        # We skip keys that start with our sentinel prefix so bookkeeping
        # entries are preserved.
        self.__dict__.update(
            {k: v for k, v in real.__dict__.items()
             if not k.startswith(_SENTINEL_PREFIX)}
        )

        # Mark as loaded AFTER the merge so that concurrent threads that
        # enter _reify simultaneously find a fully-merged __dict__.
        self.__dict__[f"{_SENTINEL_PREFIX}loaded__"] = True

        return real

    # ------------------------------------------------------------------
    # Attribute protocol
    # ------------------------------------------------------------------

    def __getattr__(self, attr: str) -> Any:
        """Reify the module and return the requested attribute.

        Python calls ``__getattr__`` only when the normal attribute lookup
        chain (instance ``__dict__``, then class MRO) has already failed.
        After :meth:`_reify` merges the real module's dict into the proxy's
        ``__dict__``, those attributes are found in ``__dict__`` and this
        method is *not* called for them again.

        Args:
            attr: The attribute name being accessed.

        Returns:
            The attribute value from the real module.

        Raises:
            AttributeError: If the real module does not have *attr*.
        """
        real = self._reify()
        try:
            return getattr(real, attr)
        except AttributeError:
            raise AttributeError(
                f"module {real.__name__!r} has no attribute {attr!r}"
            ) from None

    def __repr__(self) -> str:
        """Return a descriptive string without triggering reification.

        The repr explicitly shows whether the proxy is still deferred or
        has already been reified, which is useful when debugging.
        """
        full_name = self.__dict__.get(f"{_SENTINEL_PREFIX}full_name__", "?")
        loaded = self.__dict__.get(f"{_SENTINEL_PREFIX}loaded__", False)
        status = "loaded" if loaded else "deferred"
        return f"<_DeferredProxy {full_name!r} [{status}]>"

    def __reduce__(self) -> Any:
        """Support pickling by reifying the module first.

        :func:`pickle.dumps` calls ``__reduce__``; we reify so the real
        module (with its full state) is pickled rather than the hollow proxy.
        """
        real = self._reify()
        return real.__reduce__()


# ──────────────────────────────────────────────────────────────────────────────
# Public-facing internal functions
# ──────────────────────────────────────────────────────────────────────────────

def _lazy_compat(name: str, package: str | None = None) -> ModuleType:
    """Return a lazy-module proxy for *name* using the deferred-proxy shim.

    If *name* is already present in :data:`sys.modules` as a fully-loaded
    module, it is returned immediately — no proxy is created.  If it is
    already present as a :class:`_DeferredProxy`, the existing proxy is
    returned so callers share a single proxy object.

    This function is the shim-mode equivalent of ``importlib.util.lazy_import``
    on Python 3.15+.

    Args:
        name: Absolute or relative module name to defer.
        package: Anchor package for relative imports.  Required when *name*
            starts with ``"."``.

    Returns:
        A :class:`_DeferredProxy` (or the already-loaded real module if
        it was cached in :data:`sys.modules`).

    Raises:
        ImportError: If *name* is relative and *package* is ``None``.

    Example::

        >>> np = _lazy_compat("numpy")
        >>> type(np)
        <class 'lazyload._compat._DeferredProxy'>
        >>> np.array([1, 2])   # reification happens here
        array([1, 2])
    """
    full_name = _resolve_full_name(name, package)

    existing = sys.modules.get(full_name)

    # Already a real, fully-loaded module — no proxy needed.
    if existing is not None and not isinstance(existing, _DeferredProxy):
        return existing

    # Already a proxy in internal cache — return it so callers share one proxy per module.
    if full_name in _PROXY_CACHE:
        return _PROXY_CACHE[full_name]

    # Create a fresh proxy and store in _PROXY_CACHE.
    # sys.modules remains clean until first attribute access (_reify).
    proxy = _DeferredProxy(name, package)
    _PROXY_CACHE[full_name] = proxy
    return proxy


# ──────────────────────────────────────────────────────────────────────────────
# _LazyImportsCompat — context manager
# ──────────────────────────────────────────────────────────────────────────────

def _make_lazy_importer(original: Callable[..., ModuleType]) -> Callable[..., ModuleType]:
    """Return a replacement for ``builtins.__import__`` that defers bare imports.

    The returned callable intercepts the ``import X`` pattern (bare, absolute,
    no ``fromlist``) and routes it through :func:`_lazy_compat`.  Every other
    form — ``from X import Y``, relative imports, submodule imports like
    ``import X.Y`` — is forwarded to *original* unchanged.

    Why only bare ``import X``?
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~
    * ``from X import Y`` — Python must resolve attribute ``Y`` from module
      ``X``, which requires the module to be fully loaded.  Deferring this
      would require loading ``X`` immediately anyway, defeating the purpose.
    * ``import X.Y`` — Python expects both ``sys.modules["X"]`` and
      ``sys.modules["X.Y"]`` to be set, and the statement returns the
      top-level package ``X`` to the caller's namespace.  Correctly proxying
      the full chain requires non-trivial bookkeeping; deferring only the
      top-level ``import X`` in this case is better handled by calling
      :func:`_lazy_compat` directly.
    * Relative imports — require the calling package's context, which
      changes the ``package`` argument; forwarding these to *original*
      preserves correct resolution.

    Args:
        original: The real ``builtins.__import__`` function captured at the
            moment the context manager is entered.

    Returns:
        A drop-in replacement for ``builtins.__import__``.
    """

    def _lazy_importer(
        name: str,
        globals: dict[str, Any] | None = None,
        locals: dict[str, Any] | None = None,
        fromlist: tuple[str, ...] | list[str] = (),
        level: int = 0,
    ) -> ModuleType:
        """Intercept ``import X``; forward everything else to the real importer.

        Args:
            name: Module name (may be dotted for ``import X.Y``).
            globals: Caller's global namespace (used for relative resolution).
            locals: Caller's local namespace (unused by CPython's importer).
            fromlist: Names to import from the module (non-empty for
                ``from X import Y``).
            level: Relative import depth (0 = absolute, 1 = ``from . import``).

        Returns:
            A :class:`_DeferredProxy` for simple ``import X`` statements, or
            the result of the original importer for all other forms.
        """
        # ── Cases we leave to the original importer ──────────────────────────
        # 1. from X import Y  →  fromlist is non-empty
        # 2. Relative import  →  level > 0
        # 3. import X.Y       →  dots in name (submodule semantics)
        if fromlist or level != 0 or "." in name:
            return original(name, globals, locals, fromlist, level)

        # ── Simple bare absolute import: ``import X`` ─────────────────────────
        proxy = _lazy_compat(name)
        sys.modules[name] = proxy
        return proxy

    return _lazy_importer


class _LazyImportsCompat:
    """Context manager that makes bare ``import X`` statements lazy.

    Inside the ``with`` block, every ``import X`` statement is intercepted by
    a temporary replacement of :func:`builtins.__import__` and converted into
    a :class:`_DeferredProxy`.  When the block exits, the original importer
    is restored.

    Mechanism
    ~~~~~~~~~
    Python resolves every ``import`` statement by calling
    ``builtins.__import__``.  This is a documented, stable hook point (it is
    how ``importlib`` itself is wired in).  :class:`_LazyImportsCompat`
    replaces it on ``__enter__`` and restores it on ``__exit__``, forming a
    LIFO stack of importers that handles nesting correctly:

    .. code-block:: text

        builtins.__import__  →  <original>
                  ↓ enter outer
        builtins.__import__  →  _lazy_importer(original=<original>)
                  ↓ enter inner
        builtins.__import__  →  _lazy_importer(original=_lazy_importer(...))
                  ↓ exit inner
        builtins.__import__  →  _lazy_importer(original=<original>)
                  ↓ exit outer
        builtins.__import__  →  <original>

    Because each level saves and restores whatever was in ``builtins.__import__``
    at the moment it entered, the stack unwinds correctly regardless of
    nesting depth.

    Limitations
    ~~~~~~~~~~~
    * **``from X import Y``** is always eager — the module must be loaded to
      resolve the attribute.
    * **``import X.Y``** is always eager — submodule semantics require both
      parent and child to be registered.

    Thread safety
    ~~~~~~~~~~~~~
    ``builtins.__import__`` is a single global.  Patching it from multiple
    threads simultaneously creates a race condition.  This is an inherent
    limitation of the shim approach; the native backend (Python 3.15+) does
    not share this limitation because it uses a thread-local flag.  For
    multi-threaded code on 3.10–3.14, prefer :func:`_lazy_compat` directly
    over the context manager.

    Example::

        with _LazyImportsCompat():
            import pandas         # returns a _DeferredProxy immediately
            import requests       # returns a _DeferredProxy immediately

        # Neither pandas nor requests has been loaded yet.
        df = pandas.DataFrame()   # pandas is reified here
    """

    def __init__(self) -> None:
        """Initialise in a clean, un-entered state."""
        # Saved importer is None until __enter__; checked in __exit__
        # to give a clean error on misuse.
        self._saved_import: Callable[..., ModuleType] | None = None

    def __enter__(self) -> _LazyImportsCompat:
        """Patch ``builtins.__import__`` with the lazy interceptor.

        Returns:
            ``self``, enabling ``as`` clauses:
            ``with _LazyImportsCompat() as lz:``.
        """
        # Capture whatever is currently in builtins.__import__ — this may
        # already be a lazy importer if we are inside another
        # _LazyImportsCompat block (correct LIFO nesting).
        self._saved_import = builtins.__import__  # type: ignore[assignment]
        builtins.__import__ = _make_lazy_importer(self._saved_import)  # type: ignore[assignment]
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool | None:
        """Restore the previous ``builtins.__import__``.

        Args:
            exc_type: Exception type, if one was raised inside the block.
            exc_val: Exception instance.
            exc_tb: Traceback object.

        Returns:
            ``None`` — exceptions are never suppressed.

        Raises:
            RuntimeError: If ``__exit__`` is called without a prior
                ``__enter__``.
        """
        if self._saved_import is None:
            raise RuntimeError(
                "_LazyImportsCompat.__exit__ called before __enter__. "
                "Use this class as a context manager: "
                "'with _LazyImportsCompat(): ...'"
            )
        builtins.__import__ = self._saved_import  # type: ignore[assignment]
        self._saved_import = None
        return None


# ──────────────────────────────────────────────────────────────────────────────
# _lazy_module_compat — decorator
# ──────────────────────────────────────────────────────────────────────────────

def _lazy_module_compat(fn: _F) -> _F:
    """Decorator that defers all ``import X`` statements inside *fn*.

    Wraps *fn* so its body executes inside a :class:`_LazyImportsCompat`
    block **on the first call only**.  On subsequent calls, the wrapper
    invokes *fn* directly — all modules imported during the first call are
    already in :data:`sys.modules` (either as proxies pending reification or
    as fully-loaded modules), so re-wrapping would add overhead for no gain.

    First-call semantics
    ~~~~~~~~~~~~~~~~~~~~
    The wrapper maintains a ``_first_call`` flag in its closure (not on the
    function object, which could be mutated externally).  The sequence on the
    first invocation is:

    1. Enter :class:`_LazyImportsCompat` → patches ``builtins.__import__``.
    2. Call *fn* with the original arguments.
    3. Exit :class:`_LazyImportsCompat` → restores ``builtins.__import__``.
    4. Set ``_first_call = False``.

    Any imports executed inside *fn*'s body (and transitively inside any
    function *fn* calls, provided those functions also use bare ``import X``
    statements and run in the same thread) will be deferred.

    Limitations
    ~~~~~~~~~~~
    The same limitations as :class:`_LazyImportsCompat` apply: ``from X
    import Y`` and ``import X.Y`` remain eager.

    Thread safety
    ~~~~~~~~~~~~~
    Two threads calling the decorated function simultaneously for the first
    time will both enter :class:`_LazyImportsCompat`, creating a brief window
    where ``builtins.__import__`` is patched twice.  Because the LIFO nesting
    mechanism restores each level independently, this is safe; the only
    observable effect is that both threads defer their imports, which is the
    desired outcome.

    Args:
        fn: Any callable whose body contains bare ``import X`` statements
            that should be deferred until the function is first called.

    Returns:
        A new callable with the same name, signature, and docstring as *fn*.

    Example::

        @_lazy_module_compat
        def analyze(data):
            import scipy.stats      # deferred until analyze() is first called
            return scipy.stats.describe(data)

        # scipy.stats is NOT imported here.
        result = analyze(my_data)   # import happens during this call.
    """
    _first_call = True

    @functools.wraps(fn)
    def _wrapper(*args: Any, **kwargs: Any) -> Any:
        nonlocal _first_call
        if _first_call:
            _first_call = False
            with _LazyImportsCompat():
                return fn(*args, **kwargs)
        return fn(*args, **kwargs)

    return _wrapper  # type: ignore[return-value]
