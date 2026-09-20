"""
Overhead Benchmark Suite for lazyload.

This script measures the runtime performance overhead introduced by lazyload's
proxy mechanism compared to standard eager imports and attribute lookups.

Scenarios measured:
  1. Proxy Creation Overhead:
     Time taken to instantiate a lazy import proxy vs eager module import.
  2. First Attribute Access (Reification Overhead):
     Time taken for the first attribute access on a deferred proxy (which triggers
     real module loading and proxy reification) vs eager import + attribute access.
  3. Subsequent Attribute Access (Post-Reification Overhead):
     Time taken for attribute access on an already reified proxy vs direct module access.

Usage:
    python benchmarks/bench_overhead.py
    python benchmarks/bench_overhead.py --iterations 1000 --no-color

Requirements:
    Standard library only + lazyload package.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
import importlib
import os
from pathlib import Path
import statistics
import sys
import time
from typing import Any

# Reconfigure stdout for UTF-8 encoding on Windows console
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import lazyload
from lazyload._version import NATIVE_LAZY_IMPORTS, get_mode_description

# --- ANSI Formatting & Visual Helpers ---

def _supports_color() -> bool:
    """Return True if stdout supports ANSI color output."""
    if "NO_COLOR" in os.environ or "--no-color" in sys.argv:
        return False
    if not hasattr(sys.stdout, "isatty"):
        return False
    return sys.stdout.isatty()


USE_COLOR = _supports_color()


def _c(code: str, text: str) -> str:
    """Format text with ANSI escape sequence if color enabled."""
    if not USE_COLOR:
        return text
    return f"\033[{code}m{text}\033[0m"


def bold(text: str) -> str:
    return _c("1", text)


def dim(text: str) -> str:
    return _c("2", text)


def cyan(text: str) -> str:
    return _c("36", text)


def green(text: str) -> str:
    return _c("32", text)


def yellow(text: str) -> str:
    return _c("33", text)


def magenta(text: str) -> str:
    return _c("35", text)


def blue(text: str) -> str:
    return _c("34", text)


def _format_time(ns: float) -> str:
    """Format nanoseconds cleanly into ns, us, or ms."""
    if ns < 1_000:
        return f"{ns:.1f} ns"
    if ns < 1_000_000:
        return f"{ns / 1_000:.2f} µs"
    return f"{ns / 1_000_000:.3f} ms"


# --- Module Eviction Utilities ---

def _evict_modules(module_names: list[str]) -> dict[str, Any]:
    """Remove target modules from sys.modules and return saved references."""
    saved = {}
    for name in module_names:
        if name in sys.modules:
            saved[name] = sys.modules.pop(name)
        # Also clean up submodules if any
        submodules = [m for m in sys.modules if m.startswith(f"{name}.")]
        for sub in submodules:
            saved[sub] = sys.modules.pop(sub)
    return saved


def _restore_modules(saved: dict[str, Any]) -> None:
    """Restore saved modules back to sys.modules."""
    for name, mod in saved.items():
        sys.modules[name] = mod


# --- Benchmark Scenarios ---

TEST_MODULE = "json"
TEST_ATTR = "dumps"


def benchmark_proxy_creation(iterations: int) -> dict[str, list[float]]:
    """Measure creation time of lazy proxy vs direct import module call."""
    eager_times_ns: list[float] = []
    lazy_times_ns: list[float] = []

    # Target module for creation benchmark: "urllib.parse"
    target = "urllib.parse"

    for _ in range(iterations):
        saved = _evict_modules([target])
        try:
            t0 = time.perf_counter_ns()
            mod = importlib.import_module(target)
            t1 = time.perf_counter_ns()
            eager_times_ns.append(float(t1 - t0))
        finally:
            _restore_modules(saved)

        saved = _evict_modules([target])
        try:
            t0 = time.perf_counter_ns()
            proxy = lazyload.lazy(target)
            t1 = time.perf_counter_ns()
            lazy_times_ns.append(float(t1 - t0))
        finally:
            _restore_modules(saved)

    return {"eager": eager_times_ns, "lazy": lazy_times_ns}


def benchmark_first_access(iterations: int) -> dict[str, list[float]]:
    """Measure time to perform first attribute access (reification) vs eager import + access."""
    eager_times_ns: list[float] = []
    lazy_times_ns: list[float] = []

    target = "html.parser"
    attr = "HTMLParser"

    for _ in range(iterations):
        # 1. Eager import + access
        saved = _evict_modules([target, "html"])
        try:
            t0 = time.perf_counter_ns()
            mod = importlib.import_module(target)
            val = getattr(mod, attr)
            t1 = time.perf_counter_ns()
            eager_times_ns.append(float(t1 - t0))
        finally:
            _restore_modules(saved)

        # 2. Lazy proxy creation + first access (triggers reification)
        saved = _evict_modules([target, "html"])
        try:
            proxy = lazyload.lazy(target)
            t0 = time.perf_counter_ns()
            val = getattr(proxy, attr)
            t1 = time.perf_counter_ns()
            lazy_times_ns.append(float(t1 - t0))
        finally:
            _restore_modules(saved)

    return {"eager": eager_times_ns, "lazy": lazy_times_ns}


def benchmark_subsequent_access(iterations: int) -> dict[str, list[float]]:
    """Measure attribute lookup time on reified proxy vs real module."""
    eager_times_ns: list[float] = []
    lazy_times_ns: list[float] = []

    # Import real module
    mod = importlib.import_module("math")
    # Import reified proxy
    proxy = lazyload.lazy("math")
    _ = proxy.sqrt  # force reification

    for _ in range(iterations):
        t0 = time.perf_counter_ns()
        val1 = mod.sqrt
        t1 = time.perf_counter_ns()
        eager_times_ns.append(float(t1 - t0))

        t0 = time.perf_counter_ns()
        val2 = proxy.sqrt
        t1 = time.perf_counter_ns()
        lazy_times_ns.append(float(t1 - t0))

    return {"eager": eager_times_ns, "lazy": lazy_times_ns}


# --- Statistics Computation & Formatting ---

def compute_stats(data: list[float]) -> dict[str, float]:
    """Calculate mean, min, max, stdev for a list of nanosecond samples."""
    if not data:
        return {"mean": 0.0, "min": 0.0, "max": 0.0, "stdev": 0.0}
    m = statistics.mean(data)
    mi = min(data)
    ma = max(data)
    std = statistics.stdev(data) if len(data) > 1 else 0.0
    return {"mean": m, "min": mi, "max": ma, "stdev": std}


def print_header(iterations: int) -> None:
    """Print benchmark header and configuration box."""
    backend_badge = green("Native PEP 810") if NATIVE_LAZY_IMPORTS else yellow("Compatibility Shim")
    mode_info = get_mode_description()

    print(bold(cyan("┌──────────────────────────────────────────────────────────────┐")))
    print(bold(cyan("│                   lazyload Overhead Benchmark                │")))
    print(bold(cyan("└──────────────────────────────────────────────────────────────┘")))
    print(f"  {bold('Python Version')} : {sys.version.split()[0]} ({sys.platform})")
    print(f"  {bold('Active Backend')}: {backend_badge}")
    print(f"  {bold('Mode Info')}     : {dim(mode_info)}")
    print(f"  {bold('Iterations')}    : {cyan(str(iterations))} samples per test scenario")
    print()


def print_scenario_results(
    scenario_title: str,
    description: str,
    eager_stats: dict[str, float],
    lazy_stats: dict[str, float],
) -> None:
    """Print formatted comparison table for a single scenario."""
    print(bold(magenta(f"► {scenario_title}")))
    print(dim(f"  {description}"))
    print()

    # Table Header
    line = "├───────────────────────┼──────────────┼──────────────┼──────────────┼──────────────┤"
    top  = "┌───────────────────────┬──────────────┬──────────────┬──────────────┬──────────────┐"
    bot  = "└───────────────────────┴──────────────┴──────────────┴──────────────┴──────────────┘"

    print(cyan(top))
    print(
        cyan("│ ")
        + bold("Approach".ljust(22))
        + cyan(" │ ")
        + bold("Mean".rjust(12))
        + cyan(" │ ")
        + bold("Min".rjust(12))
        + cyan(" │ ")
        + bold("Max".rjust(12))
        + cyan(" │ ")
        + bold("StdDev".rjust(12))
        + cyan(" │")
    )
    print(cyan(line))

    # Eager Row
    e_mean = _format_time(eager_stats["mean"]).rjust(12)
    e_min  = _format_time(eager_stats["min"]).rjust(12)
    e_max  = _format_time(eager_stats["max"]).rjust(12)
    e_std  = _format_time(eager_stats["stdev"]).rjust(12)
    print(
        cyan("│ ")
        + "Eager Standard".ljust(22)
        + cyan(" │ ")
        + e_mean
        + cyan(" │ ")
        + e_min
        + cyan(" │ ")
        + e_max
        + cyan(" │ ")
        + e_std
        + cyan(" │")
    )

    # Lazy Row
    l_mean = _format_time(lazy_stats["mean"]).rjust(12)
    l_min  = _format_time(lazy_stats["min"]).rjust(12)
    l_max  = _format_time(lazy_stats["max"]).rjust(12)
    l_std  = _format_time(lazy_stats["stdev"]).rjust(12)
    print(
        cyan("│ ")
        + yellow("lazyload Proxy").ljust(22)
        + cyan(" │ ")
        + yellow(l_mean)
        + cyan(" │ ")
        + yellow(l_min)
        + cyan(" │ ")
        + yellow(l_max)
        + cyan(" │ ")
        + yellow(l_std)
        + cyan(" │")
    )

    print(cyan(bot))

    # Overhead comparison text
    mean_diff = lazy_stats["mean"] - eager_stats["mean"]
    if eager_stats["mean"] > 0:
        ratio = lazy_stats["mean"] / eager_stats["mean"]
        if ratio < 1.0:
            saved_ns = abs(mean_diff)
            verdict = green(f"⚡ {saved_ns / 1_000:.2f} µs FASTER ({1.0/ratio:.2f}x speedup)")
        else:
            overhead_ns = mean_diff
            verdict = dim(f"Δ {overhead_ns / 1_000:.2f} µs difference ({ratio:.2f}x ratio)")
    else:
        verdict = ""

    print(f"  {bold('Summary')}: {verdict}")
    print()


def main() -> int:
    """Parse CLI arguments and run benchmark scenarios."""
    parser = argparse.ArgumentParser(
        description="Benchmark overhead of lazyload proxy vs eager operations."
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=1000,
        help="Number of iterations for each benchmark scenario (default: 1000).",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable color output.",
    )
    args = parser.parse_args()

    print_header(args.iterations)

    print(dim("Warming up benchmark loops..."))
    # Warm up JIT/bytecode compilation
    _ = benchmark_proxy_creation(5)
    _ = benchmark_first_access(5)
    _ = benchmark_subsequent_access(5)
    print(dim("Warmup complete. Running benchmark suites...\n"))

    # Scenario 1: Proxy Creation
    res1 = benchmark_proxy_creation(args.iterations)
    s1_eager = compute_stats(res1["eager"])
    s1_lazy = compute_stats(res1["lazy"])
    print_scenario_results(
        "Scenario 1: Proxy Creation Overhead",
        "Measures time to register/create lazy import proxy vs eager module import",
        s1_eager,
        s1_lazy,
    )

    # Scenario 2: First Attribute Access (Reification)
    res2 = benchmark_first_access(args.iterations)
    s2_eager = compute_stats(res2["eager"])
    s2_lazy = compute_stats(res2["lazy"])
    print_scenario_results(
        "Scenario 2: First Attribute Access (Reification)",
        "Measures first attribute lookup on deferred proxy (triggers real import) vs eager import + access",
        s2_eager,
        s2_lazy,
    )

    # Scenario 3: Subsequent Attribute Access
    res3 = benchmark_subsequent_access(args.iterations)
    s3_eager = compute_stats(res3["eager"])
    s3_lazy = compute_stats(res3["lazy"])
    print_scenario_results(
        "Scenario 3: Subsequent Attribute Access (Post-Reification)",
        "Measures attribute lookup on reified proxy vs direct module lookup",
        s3_eager,
        s3_lazy,
    )

    print(bold(green("✔ Overhead benchmark completed successfully.")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
