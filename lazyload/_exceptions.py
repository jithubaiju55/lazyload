"""lazyload._exceptions — Centralised exception hierarchy for the package.

Design decision: why one file for all exceptions?
--------------------------------------------------
In a library whose whole purpose is to manipulate Python's import machinery,
errors can originate from several different places: the version-detection
layer, the deferred-proxy shim, the native backend, and the public API
functions.  Defining exceptions close to where they are raised — scattered
across ``_version.py``, ``_compat.py``, ``_native.py`` — would require every
module that *catches* an exception to know which file to import it from, and
would make it trivially easy to accidentally import the wrong class.

Centralising every exception in this one module means:

* Users need a single ``from lazyload import LazyLoadError`` to catch anything
  the package can ever raise.
* Internal modules import from one predictable location:
  ``from lazyload._exceptions import <ExceptionClass>``.
* The hierarchy is visible at a glance — no hunting through source files.
* Adding a new exception in the future requires editing only this file.

Exception hierarchy
-------------------
::

    Exception
    └── LazyLoadError                          (base for all lazyload errors)
        ├── UnsupportedPythonVersion           (interpreter too old)
        └── CircularImportError                (deferred import forms a cycle)

    Exception
    ├── ImportError
    │   └── ModuleNotFoundError  (built-in)
    └── LazyLoadError
        └── lazyload.ModuleNotFoundError       (double-inherits both paths)

The double-inheritance of :class:`LazyLoadError.ModuleNotFoundError` is the
most important structural choice: it means ``except ModuleNotFoundError`` (the
built-in) and ``except LazyLoadError`` both catch it, so existing code that
already handles ``ModuleNotFoundError`` continues to work without any changes.
"""

from __future__ import annotations

import builtins

# Capture the built-in before we shadow the name with our own class.
_BuiltinModuleNotFoundError: type[builtins.ModuleNotFoundError] = (
    builtins.ModuleNotFoundError
)


# ──────────────────────────────────────────────────────────────────────────────
# Base exception
# ──────────────────────────────────────────────────────────────────────────────

class LazyLoadError(Exception):
    """Base class for every exception raised by the lazyload package.

    All lazyload-specific exceptions inherit from :class:`LazyLoadError`, so
    a single ``except LazyLoadError`` clause is sufficient to catch *any* error
    that originates from this package.  You can also catch sub-classes
    individually when you need finer-grained control.

    When to catch this
    ~~~~~~~~~~~~~~~~~~
    Catch :class:`LazyLoadError` at application boundaries — for example, in a
    CLI entry-point or a framework plugin loader — when you want to present a
    user-friendly error for *any* lazyload failure without caring about the
    specific cause::

        try:
            import lazyload
            np = lazyload.lazy("numpy")
        except lazyload.LazyLoadError as exc:
            print(f"Lazy-import setup failed: {exc}", file=sys.stderr)
            raise SystemExit(1)

    Catch a concrete sub-class when you want to react differently to different
    failure modes (for example, re-raising :class:`UnsupportedPythonVersion`
    but recovering from :class:`CircularImportError` by falling back to an
    eager import).
    """


# ──────────────────────────────────────────────────────────────────────────────
# UnsupportedPythonVersion
# ──────────────────────────────────────────────────────────────────────────────

class UnsupportedPythonVersion(LazyLoadError):
    """Raised when lazyload is running on an interpreter version below 3.10.

    lazyload requires Python 3.10 as its minimum supported version because:

    * ``importlib.util.resolve_name`` — used to resolve relative module names —
      gained its stable public API in Python 3.9, but the import-system hooks
      that lazyload's shim relies on were not fully stabilised until 3.10.
    * Type annotation syntax used throughout the codebase (``X | Y`` union
      types, ``ParamSpec``, etc.) requires 3.10 or later.

    When is this raised?
    ~~~~~~~~~~~~~~~~~~~~
    :class:`UnsupportedPythonVersion` is raised exactly once, at the moment
    :mod:`lazyload._version` is first imported, which happens as part of the
    normal ``import lazyload`` process.  It is therefore raised *before* any
    lazy-import call can be made; it can never be raised mid-program.

    What to do
    ~~~~~~~~~~
    Upgrade your Python interpreter to 3.10 or later.  The official CPython
    downloads are at https://www.python.org/downloads/.

    If you are maintaining a library that uses lazyload and need to support
    older Pythons, add a ``python_requires = ">=3.10"`` constraint to your own
    ``pyproject.toml`` so that users on older interpreters see a clear pip
    error at install time rather than a runtime crash.

    Attributes:
    ----------
    actual_version:
        The version string of the running interpreter, e.g. ``"3.9.18"``.
    minimum_version:
        The minimum version string that lazyload requires, e.g. ``"3.10"``.

    Example:
    -------
    Running ``import lazyload`` on Python 3.9 raises::

        lazyload.UnsupportedPythonVersion:
            lazyload requires Python 3.10 or later.
            You are running Python 3.9.18.
            Please upgrade to Python 3.10+ to use this package.
            Downloads: https://www.python.org/downloads/
    """

    #: The exact version string of the running interpreter.
    actual_version: str

    #: The minimum version string required by this release of lazyload.
    minimum_version: str

    def __init__(
        self,
        actual_version: str,
        minimum_version: str = "3.10",
    ) -> None:
        """Construct the error with interpreter-version context.

        Args:
            actual_version: The version string of the running interpreter
                (e.g. ``"3.9.18"``).  Pass :data:`lazyload._version.PYTHON_VERSION`
                here when raising from internal code.
            minimum_version: The minimum version required.  Defaults to
                ``"3.10"`` and should rarely need to be overridden.
        """
        self.actual_version = actual_version
        self.minimum_version = minimum_version
        message = (
            f"\n"
            f"  lazyload requires Python {minimum_version} or later.\n"
            f"  You are running Python {actual_version}.\n"
            f"  Please upgrade to Python {minimum_version}+ to use this package.\n"
            f"  Downloads: https://www.python.org/downloads/"
        )
        super().__init__(message)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"actual_version={self.actual_version!r}, "
            f"minimum_version={self.minimum_version!r})"
        )


# ──────────────────────────────────────────────────────────────────────────────
# CircularImportError
# ──────────────────────────────────────────────────────────────────────────────

class CircularImportError(LazyLoadError):
    """Raised when a deferred import would create an unresolvable circular dependency.

    A circular import occurs when module A imports module B, and module B (or
    something B imports) imports module A in return.  Python's normal import
    system partially handles this by providing the *partially-initialised*
    module A to B; the assumption is that A has already defined everything B
    needs by the time the circular reference is evaluated.

    Lazy imports break this assumption.  If A lazily defers B, and B is reified
    (i.e. actually imported) *before* A has finished initialising, then at the
    moment B tries to import A it may find an empty or half-populated module —
    or, in the worst case, it may trigger a fresh import of A that recursively
    triggers the deferral again, producing infinite recursion.

    lazyload detects this situation during reification and raises
    :class:`CircularImportError` rather than silently producing incorrect
    behaviour.

    When is this raised?
    ~~~~~~~~~~~~~~~~~~~~
    This error is raised at the moment a deferred proxy is *reified* (i.e.
    when the first attribute access triggers the real import) if lazyload
    detects that the import chain leads back to a module that is already
    mid-import.

    What to do
    ~~~~~~~~~~
    You have two practical options:

    1. **Move the import inside the function that uses it** — this is often the
       correct fix because it delays the import until after both modules have
       finished initialising::

           # Instead of lazy-importing at module level:
           # heavy = lazyload.lazy("heavy_module")   # might be circular

           # Do this inside the function that needs it:
           def process():
               import heavy_module
               return heavy_module.run()

    2. **Restructure the module graph** — extract the shared dependency into a
       third module that neither A nor B imports circularly.

    Attributes:
    ----------
    module_name:
        The name of the module whose deferred import triggered the cycle.
    cycle:
        An ordered list of module names describing the full circular path.
        For example, ``["pkg.a", "pkg.b", "pkg.c", "pkg.a"]`` means
        *a → b → c → a*.  May be empty if the cycle could not be reconstructed.

    Example:
    -------
    ::

        try:
            result = my_lazy_module.some_function()
        except lazyload.CircularImportError as exc:
            print(f"Circular import detected: {' → '.join(exc.cycle)}")
    """

    #: The name of the module whose reification triggered the cycle.
    module_name: str

    #: Ordered list of module names forming the cycle, including the repeated
    #: first element at the end.  E.g. ``["a", "b", "a"]``.
    cycle: list[str]

    def __init__(
        self,
        module_name: str,
        cycle: list[str] | None = None,
    ) -> None:
        """Construct the error with the offending module name and cycle path.

        Args:
            module_name: The name of the module that triggered the cycle.
            cycle: An ordered list of module names forming the circular path,
                with the first module repeated at the end so the cycle is
                visually obvious.  Pass ``None`` if the cycle cannot be
                reconstructed (a generic message is used instead).
        """
        self.module_name = module_name
        self.cycle = cycle or []

        if self.cycle:
            arrow_path = " \N{RIGHTWARDS ARROW} ".join(self.cycle)
            detail = f"Circular dependency path: {arrow_path}"
        else:
            detail = (
                "The full cycle path could not be reconstructed. "
                "Enable verbose logging for details."
            )

        message = (
            f"Deferred import of {module_name!r} created an unresolvable "
            f"circular dependency.\n"
            f"  {detail}\n"
            f"  Fix: move the import inside the function that needs it, or "
            f"restructure the module graph to eliminate the cycle."
        )
        super().__init__(message)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"module_name={self.module_name!r}, "
            f"cycle={self.cycle!r})"
        )


# ──────────────────────────────────────────────────────────────────────────────
# ModuleNotFoundError
# ──────────────────────────────────────────────────────────────────────────────

class ModuleNotFoundError(LazyLoadError, _BuiltinModuleNotFoundError):  # type: ignore[misc, valid-type]
    """Raised when a lazily-deferred module does not exist when finally accessed.

    This exception is raised during *reification* — that is, at the moment you
    first access any attribute on a lazy proxy — if the underlying module
    cannot be found on :data:`sys.path`.

    Why not just use the built-in ``ModuleNotFoundError``?
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    :class:`lazyload.ModuleNotFoundError` *is* the built-in
    ``ModuleNotFoundError``: it inherits from both :class:`LazyLoadError` and
    Python's :class:`builtins.ModuleNotFoundError`.  This double inheritance
    means that all of the following ``except`` clauses will catch it:

    * ``except ModuleNotFoundError`` — unchanged existing code still works.
    * ``except ImportError`` — it is still an import error.
    * ``except LazyLoadError`` — it is a lazyload error.

    Behaviour difference from an eager import
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    With an eager ``import missing_module``, Python raises
    ``ModuleNotFoundError`` immediately at the ``import`` statement.  With
    lazy loading, the proxy is created and placed in :data:`sys.modules`
    without error.  The ``ModuleNotFoundError`` is only raised when the proxy
    is *reified* — which may be much later, in a completely different part of
    the code.

    Developers should be aware of this delayed error site when debugging.  The
    exception's ``__cause__`` will carry the original ``ModuleNotFoundError``
    from the real import attempt so the full context is preserved.

    What to do
    ~~~~~~~~~~
    * Check that the package is installed: ``pip show <package-name>``.
    * Verify the exact module name (spelling, capitalisation).
    * If the module is an optional dependency, guard the usage::

          import lazyload

          _HAS_NUMPY = True
          np = lazyload.lazy("numpy")

          def compute(data):
              try:
                  return np.array(data)
              except lazyload.ModuleNotFoundError:
                  raise RuntimeError(
                      "numpy is required for compute(). "
                      "Install it with: pip install numpy"
                  ) from None

    Attributes:
    ----------
    module_name:
        The name of the module that could not be found.

    Example:
    -------
    ::

        np = lazyload.lazy("numpyy")   # typo — no error yet
        np.array([1, 2, 3])            # ModuleNotFoundError raised HERE
    """

    #: The name of the module that could not be found.
    module_name: str

    def __init__(
        self,
        module_name: str,
        msg: str | None = None,
    ) -> None:
        """Construct the error with the missing module's name.

        Args:
            module_name: The fully-qualified name of the module that could
                not be found (e.g. ``"numpy"`` or ``"mypackage.utils"``).
            msg: Optional override for the full error message.  When omitted,
                a standard message is generated from *module_name*.
        """
        self.module_name = module_name
        message = msg or (
            f"No module named {module_name!r}.\n"
            f"  The module was registered as a lazy import but could not be "
            f"found when first accessed.\n"
            f"  Check that {module_name!r} is installed: "
            f"pip show {module_name.split('.')[0]}"
        )
        # Call both parent __init__ paths.  LazyLoadError → Exception takes
        # the message; _BuiltinModuleNotFoundError sets .name for compatibility.
        super().__init__(message)
        self.name = module_name        # matches built-in ModuleNotFoundError.name

    def __repr__(self) -> str:
        return f"{type(self).__name__}(module_name={self.module_name!r})"


# ──────────────────────────────────────────────────────────────────────────────
# Public re-exports
# ──────────────────────────────────────────────────────────────────────────────

__all__: list[str] = [
    "CircularImportError",
    "LazyLoadError",
    "ModuleNotFoundError",
    "UnsupportedPythonVersion",
]
