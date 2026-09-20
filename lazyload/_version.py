"""lazyload._version — Runtime Python-version detection and capability flags.

Why version detection matters for lazyload
------------------------------------------
Lazy imports are not a single monolithic feature — Python has grown support for
them incrementally, and the *mechanism* that lazyload must use differs
significantly depending on the interpreter version the user is running:

**Python < 3.10 (unsupported)**
    The ``importlib.util`` module lacks several APIs that lazyload relies on,
    and ``sys.version_info``-based guards elsewhere in CPython make it
    impossible to replicate the required behaviour faithfully.  Rather than
    silently delivering incorrect results, lazyload raises an explicit
    :class:`RuntimeError` at import time with a clear upgrade message.

**Python 3.10 – 3.14 (compatibility-shim mode)**
    Lazy loading is achieved through a combination of
    :func:`importlib.util.LazyLoader`, a custom :class:`importlib.abc.MetaPathFinder`
    installed into :data:`sys.meta_path`, and careful shadowing of
    :func:`builtins.__import__` inside context managers.  All three public API
    surfaces (``lazy()``, ``lazy_imports``, ``lazy_module``) are implemented
    purely in Python, with no dependency on any C extension or standard-library
    function added after 3.10.

**Python 3.15+ (native mode)**
    `PEP 690 <https://peps.python.org/pep-0690/>`_ introduced a first-class
    lazy import mechanism into CPython itself.  When running on 3.15 or later,
    lazyload delegates directly to the built-in machinery rather than
    maintaining its own shim.  This path is faster, lighter, and immune to
    edge cases in the shim that arise from CPython internals changing between
    minor releases.

By centralising all version logic here, every other internal module imports a
single set of boolean flags and never duplicates a ``sys.version_info``
comparison.  This makes it trivial to audit compatibility, update thresholds
as new CPython releases land, and write tests that exercise both code paths.
"""

from __future__ import annotations

import sys
from typing import Literal

# ──────────────────────────────────────────────────────────────────────────────
# Raw version tuple — use this for comparisons, not the string form.
# ──────────────────────────────────────────────────────────────────────────────
_VERSION_INFO: tuple[int, int, int] = (
    sys.version_info.major,
    sys.version_info.minor,
    sys.version_info.micro,
)

# ──────────────────────────────────────────────────────────────────────────────
# Minimum-version guard — fail loudly and early.
# ──────────────────────────────────────────────────────────────────────────────
if _VERSION_INFO < (3, 10, 0):
    raise RuntimeError(
        "\n"
        "╔══════════════════════════════════════════════════════════════════╗\n"
        "║  lazyload requires Python 3.10 or later.                        ║\n"
        f"║  You are running Python {sys.version.split()[0]:<41}║\n"
        "║                                                                  ║\n"
        "║  Please upgrade your interpreter:                                ║\n"
        "║    https://www.python.org/downloads/                             ║\n"
        "╚══════════════════════════════════════════════════════════════════╝\n"
    )

# ──────────────────────────────────────────────────────────────────────────────
# Public capability flags
# ──────────────────────────────────────────────────────────────────────────────

#: ``True`` when running on Python 3.15 or above, where CPython ships a native
#: lazy-import mechanism (PEP 690).  lazyload will delegate to that built-in
#: machinery rather than maintaining its own shim.
NATIVE_LAZY_IMPORTS: bool = _VERSION_INFO >= (3, 15, 0)

#: ``True`` when running on Python 3.10 – 3.14 inclusive.  lazyload uses its
#: own compatibility shim built on :mod:`importlib` hooks and
#: :func:`builtins.__import__` overrides.
SHIM_REQUIRED: bool = (3, 10, 0) <= _VERSION_INFO < (3, 15, 0)

# Sanity check — exactly one mode must be active.
assert NATIVE_LAZY_IMPORTS ^ SHIM_REQUIRED, (
    "lazyload._version: internal inconsistency — "
    "exactly one of NATIVE_LAZY_IMPORTS or SHIM_REQUIRED must be True."
)

# ──────────────────────────────────────────────────────────────────────────────
# Human-readable version strings
# ──────────────────────────────────────────────────────────────────────────────

#: The lazyload package version string.
__version__: str = "0.1.0"

#: The running Python version as a compact ``"major.minor.micro"`` string,
#: e.g. ``"3.12.3"``.  Useful for logging, ``--version`` output, and error
#: messages throughout the package.
PYTHON_VERSION: str = "{0}.{1}.{2}".format(*_VERSION_INFO)

#: The active operating mode as a short identifier.
#: Either ``"native"`` (Python ≥ 3.15) or ``"shim"`` (Python 3.10 – 3.14).
MODE: Literal["native", "shim"] = "native" if NATIVE_LAZY_IMPORTS else "shim"


# ──────────────────────────────────────────────────────────────────────────────
# Public helper
# ──────────────────────────────────────────────────────────────────────────────

def get_mode_description() -> str:
    """Return a plain-English description of lazyload's active operating mode.

    This is intended for debugging, ``--version`` CLI output, and test
    assertions.  It is not part of the stable public API.

    Returns:
        A multi-line string describing the detected Python version, the
        active mode (native or shim), and a brief explanation of what that
        means for the user.

    Example::

        >>> from lazyload._version import get_mode_description
        >>> print(get_mode_description())
        lazyload 0.1.0 — operating in SHIM mode
          Python version : 3.12.3
          Mode           : shim
          Reason         : Python 3.10–3.14 detected; using the importlib-based
                           compatibility shim to provide lazy import behaviour.
          Upgrade path   : Upgrade to Python 3.15+ to use native lazy imports.
    """
    if NATIVE_LAZY_IMPORTS:
        mode_line = "operating in NATIVE mode"
        reason = (
            "Python 3.15+ detected; delegating directly to CPython's built-in\n"
            "          lazy-import machinery (PEP 690).  No shim is loaded."
        )
        upgrade = "You are already on the best available path."
    else:
        mode_line = "operating in SHIM mode"
        reason = (
            "Python 3.10–3.14 detected; using the importlib-based\n"
            "          compatibility shim to provide lazy import behaviour."
        )
        upgrade = "Upgrade to Python 3.15+ to use native lazy imports."

    return (
        f"lazyload {__version__} — {mode_line}\n"
        f"  Python version : {PYTHON_VERSION}\n"
        f"  Mode           : {MODE}\n"
        f"  Reason         : {reason}\n"
        f"  Upgrade path   : {upgrade}"
    )
