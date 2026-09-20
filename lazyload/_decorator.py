"""lazyload._decorator — implementation of the :func:`lazy_module` decorator.

Public surface
--------------
lazy_module(fn)
    A decorator that wraps *fn* so that every ``import`` executed during
    the *first* invocation of the function is deferred via the lazy-import
    machinery.  Subsequent calls are unaffected; the modules are already
    resolved and cached in ``sys.modules``.

Example::

    @lazy_module
    def process(data):
        import heavy_lib          # deferred until process() is first called
        return heavy_lib.run(data)

This module is internal.  Import :func:`lazyload.lazy_module` instead.
"""

from __future__ import annotations

from collections.abc import Callable
import functools
from typing import Any, TypeVar

from lazyload._version import NATIVE_LAZY_IMPORTS, SHIM_REQUIRED  # noqa: F401

_F = TypeVar("_F", bound=Callable[..., Any])


def lazy_module(fn: _F) -> _F:
    """Decorate *fn* so its internal imports are deferred until first call.

    Args:
        fn: Any callable whose body contains ``import`` statements that
            should be deferred.

    Returns:
        A wrapper with the same signature as *fn*.

    Raises:
        NotImplementedError: Until the real implementation is written.
    """

    @functools.wraps(fn)
    def _wrapper(*args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    return _wrapper  # type: ignore[return-value]
