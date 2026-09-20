"""lazyload._api — Version-aware dispatcher for the public API.

Why this file exists
--------------------
The public symbols ``lazy``, ``lazy_imports``, and ``lazy_module`` need to
work identically on every supported Python version, but the *mechanism* that
makes them work is completely different between Python 3.10–3.14 (the
deferred-proxy shim in :mod:`lazyload._compat`) and Python 3.15+ (the native
CPython machinery in :mod:`lazyload._native`).

This file is the single place where that routing decision is made.

Why not inline this into ``__init__.py``?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
``__init__.py`` is the *public face* of the package — its job is to declare
what the package exports and document it for users.  Mixing version-dispatch
logic into it would blur that responsibility and make both files harder to
read, test, and maintain.  Keeping the dispatch here means:

* ``__init__.py`` stays a clean, one-paragraph reception desk.
* This module can be imported independently in tests to verify routing.
* Adding a third backend in the future (e.g. a C-extension fast path) only
  requires editing this file and adding the new backend module.

How routing works — zero call-time overhead
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The routing decision happens **once**, at the moment this module is first
imported, not at each individual call.  The ``if NATIVE_LAZY_IMPORTS`` branch
at module level binds ``_backend_lazy``, ``_backend_lazy_imports``, and
``_backend_lazy_module`` to the appropriate backend callables.  Each public
function then delegates to those module-level names with a single Python call
— no ``if`` branch, no ``getattr``, no lookup in any dispatch table.

This module is **internal**.  Users should import from :mod:`lazyload`
directly, not from ``lazyload._api``.
"""

from __future__ import annotations

from collections.abc import Callable
from types import ModuleType
from typing import TYPE_CHECKING, Any, TypeVar

from lazyload._version import NATIVE_LAZY_IMPORTS

if TYPE_CHECKING:
    from lazyload._types import LazyImportsProtocol

_F = TypeVar("_F", bound=Callable[..., Any])
_backend_lazy_imports: Callable[[], LazyImportsProtocol]

# ──────────────────────────────────────────────────────────────────────────────
# Backend selection  — happens once, at import time
# ──────────────────────────────────────────────────────────────────────────────

if NATIVE_LAZY_IMPORTS:
    # Python 3.15+ — delegate to CPython's own PEP 810 machinery.
    from lazyload._native import _lazy_module_native as _backend_lazy_module
    from lazyload._native import _lazy_native as _backend_lazy
    from lazyload._native import _LazyImportsNative as _backend_lazy_imports
else:
    # Python 3.10–3.14 — delegate to the deferred-proxy shim.
    from lazyload._compat import _lazy_compat as _backend_lazy
    from lazyload._compat import _lazy_module_compat as _backend_lazy_module
    from lazyload._compat import _LazyImportsCompat as _backend_lazy_imports


# ──────────────────────────────────────────────────────────────────────────────
# Public API functions
# ──────────────────────────────────────────────────────────────────────────────


def lazy(name: str, package: str | None = None) -> ModuleType:
    """Defer loading of a module until its first attribute is accessed.

    Call ``lazy()`` anywhere you would write a regular ``import`` statement
    when you want to postpone the cost of loading that module.  The returned
    object looks and behaves exactly like the real module — you can pass it
    around, assign it, and use it as normal — but the actual file-system read,
    bytecode compilation, and module body execution are all held back until the
    first time you touch any attribute on it.

    This is the simplest and most explicit form of lazy loading.  Use it when
    you want to defer a single, named module, and you know the module's name
    at the point where you write the call.

    Parameters
    ----------
    name:
        The module to defer.  Accepts the same forms as a regular ``import``
        statement: a plain name (``"numpy"``), a dotted path
        (``"email.mime.text"``), or a relative name starting with dots
        (``".utils"``).
    package:
        Required only when *name* is a relative import (starts with ``"."``).
        Pass the ``__name__`` attribute of the calling module so that the
        relative path can be resolved correctly.  Ignored for absolute names.

    Returns:
    -------
    ModuleType
        A module proxy.  On Python 3.15+ this is CPython's native
        ``LazyModuleProxy``; on 3.10–3.14 it is a :class:`~lazyload._compat.
        _DeferredProxy`.  Either way it quacks exactly like the real module
        once any attribute is accessed.

    Raises:
    ------
    ImportError
        If *name* is relative (starts with ``"."``) and *package* is ``None``.
    ModuleNotFoundError
        If the module cannot be found on :data:`sys.path`.  The error is
        raised at the moment you first access an attribute, not when you call
        ``lazy()``.

    Example:
    -------
    Imagine you have a CLI tool that only needs ``pandas`` when the user runs
    a specific sub-command.  Instead of paying the full ``pandas`` import cost
    at startup for every invocation, you write::

        import lazyload

        pd = lazyload.lazy("pandas")  # ← no disk I/O here


        def summarise(path):
            # pandas is imported here, on the first call to summarise()
            return pd.read_csv(path).describe()

    The user launching ``--help`` or any other sub-command never waits for
    ``pandas`` to load.  Only callers of ``summarise()`` pay the import cost,
    and they pay it exactly once.
    """
    return _backend_lazy(name, package)


def lazy_imports() -> Any:
    """Return a context manager that defers every ``import X`` in its block.

    Use ``lazy_imports()`` when you want to defer a *group* of imports that
    are written together in the same place.  Inside the ``with`` block, every
    bare ``import X`` statement is intercepted and converted to a lazy proxy;
    nothing is actually loaded from the file system until one of the imported
    names is first used.

    This is the most readable form of lazy loading for modules you import at
    the top of a function or method body, because it lets you keep the
    ``import`` statements exactly as you would write them normally — you just
    wrap them in a ``with`` block.

    .. note::

        ``from X import Y`` statements are **always eager** — because Python
        must load module ``X`` before it can resolve attribute ``Y``, there is
        no way to defer them without loading the module anyway.  Use
        :func:`lazy` for those cases instead.

    Parameters
    ----------
    None

    Returns:
    -------
    contextmanager
        A context manager.  On Python 3.15+ this wraps ``importlib.lazy()``;
        on 3.10–3.14 it temporarily patches ``builtins.__import__`` with a
        lazy interceptor and restores it on exit.  Both behave identically
        from the caller's perspective.

    Example:
    -------
    Suppose you have a data-processing function that relies on several heavy
    scientific libraries, but those libraries are only an optional dependency
    that not all users install.  You can defer all of their imports to the
    moment the function is first called::

        import lazyload

        with lazyload.lazy_imports():
            import numpy as np
            import scipy.signal as signal
            import matplotlib.pyplot as plt


        def plot_spectrum(data):
            # numpy, scipy, and matplotlib are loaded here on first use,
            # not at the top of the module.
            freqs = np.fft.rfftfreq(len(data))
            plt.plot(freqs, np.abs(signal.periodogram(data)[1]))
            plt.show()

    If the user never calls ``plot_spectrum()``, none of those three libraries
    are ever loaded.
    """
    return _backend_lazy_imports()


def lazy_module(fn: _F) -> _F:
    """Decorator that defers every ``import X`` inside a function until first call.

    Apply ``@lazy_module`` to any function whose body contains ``import``
    statements that you want to defer.  The decorated function behaves
    identically to the original — same name, same signature, same return value
    — but on its *first* invocation the import statements inside it are
    silently converted to lazy proxies.  On all subsequent calls, the function
    runs exactly as written; the modules are already cached in
    :data:`sys.modules` and the import statements resolve instantly.

    This is the most convenient form of lazy loading for functions that import
    heavy dependencies internally, because it requires no change to the code
    inside the function body — you simply add one decorator line.

    .. note::

        Like :func:`lazy_imports`, this decorator only intercepts bare
        ``import X`` statements.  ``from X import Y`` statements remain eager.

    Parameters
    ----------
    fn:
        Any callable.  The decorator preserves its ``__name__``,
        ``__doc__``, ``__annotations__``, and ``__wrapped__`` attributes via
        :func:`functools.wraps`, so introspection tools and type-checkers see
        the original function unchanged.

    Returns:
    -------
    Callable
        A wrapper with the same type signature as *fn*.

    Example:
    -------
    Consider a module that converts documents using a heavyweight library.
    Without ``@lazy_module``, every program that imports your module pays the
    full import cost of the library — even if it never calls
    ``convert_document``::

        import lazyload


        @lazyload.lazy_module
        def convert_document(path, fmt="pdf"):
            import weasyprint  # ← deferred until convert_document()
            import pypandoc  # ← deferred until convert_document()

            if fmt == "pdf":
                return weasyprint.HTML(path).write_pdf()
            return pypandoc.convert_file(path, fmt)

    The first time ``convert_document()`` is called, ``weasyprint`` and
    ``pypandoc`` are imported.  Every subsequent call finds them already
    in :data:`sys.modules` and runs at full speed.  Any program that imports
    your module but never calls ``convert_document()`` pays zero import cost
    for those libraries.
    """
    return _backend_lazy_module(fn)
