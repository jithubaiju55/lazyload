"""lazyload._context — implementation of the :class:`lazy_imports` context manager.

Public surface
--------------
lazy_imports()
    A context manager that intercepts every ``import`` statement executed
    inside its ``with`` block and converts each one into a lazy proxy,
    deferring the real I/O until the first attribute access.

Example::

    with lazy_imports():
        import numpy as np  # not yet imported
        import pandas as pd  # not yet imported

    df = pd.DataFrame()  # pandas is imported here, on first use

This module is internal.  Import :class:`lazyload.lazy_imports` instead.
"""

from __future__ import annotations

from contextlib import AbstractContextManager
from types import TracebackType

from lazyload._version import NATIVE_LAZY_IMPORTS, SHIM_REQUIRED  # noqa: F401


class lazy_imports(AbstractContextManager["lazy_imports"]):  # noqa: N801
    """Context manager that makes every ``import`` inside the block lazy.

    Yields:
        The manager instance itself (for potential future use in ``as`` clauses).

    Raises:
        NotImplementedError: Until the real implementation is written.
    """

    def __enter__(self) -> lazy_imports:
        """Install the lazy import hook."""
        raise NotImplementedError

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool | None:
        """Remove the lazy import hook."""
        raise NotImplementedError
