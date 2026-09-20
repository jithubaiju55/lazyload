"""lazyload._types — Shared protocols and type definitions for lazyload."""

from __future__ import annotations

from types import TracebackType
from typing import Protocol, runtime_checkable


@runtime_checkable
class LazyImportsProtocol(Protocol):
    """Protocol defining the context manager interface for lazy import scopes."""

    def __enter__(self) -> LazyImportsProtocol:
        ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool | None:
        ...
