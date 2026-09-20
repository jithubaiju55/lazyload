"""lazyload — Lazy imports for Python 3.10 through 3.15+, one API, every version.

This package gives developers a single, ergonomic interface for deferring module
imports until the moment they are first used.  Whether you are trimming the startup
time of a CLI, preventing circular-import tangles in a large library, or writing
optional-dependency guards without boilerplate, lazyload exposes three complementary
tools that cover every common pattern cleanly and without runtime overhead once the
import is resolved:

* :func:`lazy` — defer a single named module.
* :func:`lazy_imports` — defer a group of imports inside a ``with`` block.
* :func:`lazy_module` — defer imports inside a function until its first call.

On Python 3.15+ the package delegates to CPython's native PEP 810 machinery.
On Python 3.10–3.14 a transparent deferred-proxy shim provides identical behaviour.
"""

from __future__ import annotations

from lazyload._api import lazy, lazy_imports, lazy_module
from lazyload._exceptions import (
    CircularImportError,
    LazyLoadError,
    ModuleNotFoundError,
    UnsupportedPythonVersion,
)

__version__: str = "0.1.0"
__author__: str = "Jithu"
__license__: str = "MIT"

__all__: list[str] = [
    # Core API
    "lazy",
    "lazy_imports",
    "lazy_module",
    # Exceptions
    "LazyLoadError",
    "UnsupportedPythonVersion",
    "CircularImportError",
    "ModuleNotFoundError",
]
