"""lazyload._lazy — core implementation of the :func:`lazy` function.

Public surface
--------------
lazy(name, package=None)
    Return a proxy object that behaves exactly like the named module but
    defers the actual ``importlib.import_module`` call until the first
    attribute access on the proxy.

This module is internal.  Import :func:`lazyload.lazy` instead.
"""

from __future__ import annotations

from types import ModuleType

from lazyload._version import NATIVE_LAZY_IMPORTS, SHIM_REQUIRED  # noqa: F401


def lazy(name: str, package: str | None = None) -> ModuleType:
    """Return a lazily-imported proxy for *name*.

    Args:
        name: Absolute or relative module name (e.g. ``"os.path"``).
        package: Anchor package for relative imports.  Required when *name*
            starts with a dot; ignored otherwise.

    Returns:
        A :class:`~types.ModuleType` proxy that imports *name* on first use.

    Raises:
        NotImplementedError: Until the real implementation is written.
    """
    raise NotImplementedError
