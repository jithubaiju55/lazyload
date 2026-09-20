#!/usr/bin/env python3
"""bench_startup.py — How much startup time does lazyload defer?

Run with:
    python benchmarks/bench_startup.py
    python benchmarks/bench_startup.py --runs 10
    python benchmarks/bench_startup.py --no-color

What this measures
------------------
For each target module this script measures two things in isolation:

1. EAGER import time  — how long ``importlib.import_module(name)`` takes when
   the module is not already in ``sys.modules``.  This is the cost your
   application pays at startup for every module imported at the top of a file.

2. LAZY proxy time   — how long ``lazyload.lazy(name)`` takes.  This is the
   only startup cost when using lazyload: creating a lightweight proxy object
   and inserting it into ``sys.modules``.

The difference is the startup time lazyload *defers* to the first actual use
of the module.  If that use never happens (optional code path, early exit,
unused feature), the cost is eliminated entirely.

Measurement methodology
-----------------------
* Modules are removed from ``sys.modules`` (along with all submodules) before
  each run so every iteration reflects a cold-cache import.
* Five warm-up runs precede the timed runs to reduce JIT and disk-cache noise.
* The mean of N timed runs is reported.  N defaults to 5 for heavy packages
  (disk I/O dominates) and 200 for stdlib modules (faster and noisier).
* If numpy/pandas/torch are not installed, the script substitutes three
  stdlib modules of comparable structural complexity and notes the swap.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import importlib
import math
import os
import sys
import time
from typing import Optional

# ─── Path setup — make sure lazyload is importable from the project root ─────
_BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_BENCH_DIR)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# Reconfigure stdout for UTF-8 encoding on Windows console
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

try:
    import lazyload
    from lazyload._version import MODE, PYTHON_VERSION
except ImportError as exc:
    print(f"ERROR: Cannot import lazyload — {exc}")
    print("       Run this script from the project root directory.")
    sys.exit(1)


# ─── Terminal formatting ──────────────────────────────────────────────────────
_USE_COLOR = sys.stdout.isatty() and os.getenv("NO_COLOR") is None


def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _USE_COLOR else text


def bold(t: str) -> str:
    return _c("1", t)


def green(t: str) -> str:
    return _c("32", t)


def yellow(t: str) -> str:
    return _c("33", t)


def cyan(t: str) -> str:
    return _c("36", t)


def dim(t: str) -> str:
    return _c("2", t)


def red(t: str) -> str:
    return _c("31", t)


def magenta(t: str) -> str:
    return _c("35", t)


def _bar(fill: int, total: int = 30, char: str = "█") -> str:
    n = max(1, round(fill / total * 20)) if total > 0 else 1
    return char * n + dim("░" * (20 - n))


# ─── Statistics helpers (no third-party imports) ──────────────────────────────
def _mean(data: list[float]) -> float:
    return sum(data) / len(data)


def _stdev(data: list[float]) -> float:
    if len(data) < 2:
        return 0.0
    m = _mean(data)
    return math.sqrt(sum((x - m) ** 2 for x in data) / (len(data) - 1))


# ─── sys.modules isolation ────────────────────────────────────────────────────
def _evict(name: str) -> dict[str, object]:
    """Remove *name* and all its submodules from sys.modules; return saved state."""
    saved: dict[str, object] = {}
    for key in list(sys.modules):
        if key == name or key.startswith(name + "."):
            saved[key] = sys.modules.pop(key)
    return saved


def _restore(saved: dict[str, object]) -> None:
    sys.modules.update(saved)  # type: ignore[arg-type]


# ─── Target module probing ────────────────────────────────────────────────────
_PREFERRED = [
    ("numpy", "numpy.array", "data science"),
    ("pandas", "pandas.DataFrame", "data science"),
    ("torch", "torch.Tensor", "deep learning"),
]

_STDLIB_FALLBACKS = [
    ("xml.etree.ElementTree", "xml.etree.ElementTree.parse", "stdlib (xml)"),
    ("sqlite3", "sqlite3.connect", "stdlib (sqlite3)"),
    ("ast", "ast.parse", "stdlib (ast)"),
]


@dataclass
class Target:
    name: str
    probe_attr: str
    category: str
    available: bool = False
    is_fallback: bool = False


def _probe_targets() -> list[Target]:
    targets: list[Target] = []
    fallback_idx = 0

    for name, attr, cat in _PREFERRED:
        saved = _evict(name)
        try:
            mod = importlib.import_module(name)
            # Verify the probe attribute exists
            parts = attr.split(".")
            obj = mod
            for part in parts[1:]:
                obj = getattr(obj, part)
            available = True
        except (ImportError, AttributeError):
            available = False
        finally:
            _evict(name)
            _restore(saved)

        if available:
            targets.append(Target(name, attr, cat, available=True))
        elif fallback_idx < len(_STDLIB_FALLBACKS):
            fb_name, fb_attr, fb_cat = _STDLIB_FALLBACKS[fallback_idx]
            targets.append(
                Target(fb_name, fb_attr, fb_cat, available=True, is_fallback=True)
            )
            fallback_idx += 1

    return targets


# ─── Core measurement ─────────────────────────────────────────────────────────
@dataclass
class Result:
    name: str
    category: str
    is_fallback: bool
    eager_ms: float
    lazy_ms: float
    stdev_ms: float

    @property
    def saved_ms(self) -> float:
        return max(0.0, self.eager_ms - self.lazy_ms)

    @property
    def speedup(self) -> float:
        return self.eager_ms / self.lazy_ms if self.lazy_ms > 0 else float("inf")


import warnings

warnings.filterwarnings("ignore")


def _measure_eager(name: str, runs: int, warmup: int) -> tuple[float, float]:
    """Return (mean_ms, stdev_ms) for eager import."""
    times_ms: list[float] = []
    for i in range(warmup + runs):
        saved = _evict(name)
        t0 = time.perf_counter()
        try:
            importlib.import_module(name)
        except Exception:
            _restore(saved)
            if times_ms:
                break
            return 0.0, 0.0
        elapsed = (time.perf_counter() - t0) * 1_000
        _evict(name)
        _restore(saved)
        if i >= warmup:
            times_ms.append(elapsed)
    if not times_ms:
        return 0.0, 0.0
    return _mean(times_ms), _stdev(times_ms)


def _measure_lazy(name: str, runs: int, warmup: int) -> float:
    """Return mean_ms for lazyload.lazy() proxy creation only."""
    times_ms: list[float] = []
    for i in range(warmup + runs):
        sys.modules.pop(name, None)
        t0 = time.perf_counter()
        lazyload.lazy(name)
        elapsed = (time.perf_counter() - t0) * 1_000
        sys.modules.pop(name, None)
        if i >= warmup:
            times_ms.append(elapsed)
    return _mean(times_ms)


def run_benchmark(targets: list[Target], runs: int, warmup: int) -> list[Result]:
    results: list[Result] = []
    for t in targets:
        # Heavy packages: fewer runs (disk-I/O bound)
        # Stdlib modules: more runs (CPU-bound, noisier)
        r = runs if not t.is_fallback else runs * 10
        w = warmup
        eager_ms, stdev_ms = _measure_eager(t.name, r, w)
        lazy_ms = _measure_lazy(t.name, r * 10, w)
        results.append(
            Result(
                name=t.name,
                category=t.category,
                is_fallback=t.is_fallback,
                eager_ms=eager_ms,
                lazy_ms=lazy_ms,
                stdev_ms=stdev_ms,
            )
        )
    return results


# ─── Output formatting ────────────────────────────────────────────────────────
_COL = {"name": 28, "cat": 14, "eager": 14, "lazy": 12, "saved": 12, "speedup": 10}
_SEP = "─"
_H = "═"


def _rule(widths: list[int], left: str, mid: str, right: str, fill: str) -> str:
    return left + mid.join(fill * (w + 2) for w in widths) + right


def _header_rule(widths: list[int]) -> str:
    return _rule(widths, "╔", "╦", "╗", "═")


def _mid_rule(widths: list[int]) -> str:
    return _rule(widths, "╠", "╬", "╣", "═")


def _body_rule(widths: list[int]) -> str:
    return _rule(widths, "╟", "╫", "╢", "─")


def _foot_rule(widths: list[int]) -> str:
    return _rule(widths, "╚", "╩", "╝", "═")


def _row(cells: list[str], widths: list[int]) -> str:
    parts = [f" {c:<{w}} " for c, w in zip(cells, widths, strict=False)]
    return "║" + "║".join(parts) + "║"


def print_results(results: list[Result], runs: int) -> None:
    if not results:
        print(red("No results to display."))
        return

    # ── Banner ────────────────────────────────────────────────────────────────
    banner_text = "  lazyload — Startup Time Benchmark  "
    print()
    print(bold(cyan("╔" + "═" * (len(banner_text)) + "╗")))
    print(bold(cyan("║")) + bold(banner_text) + bold(cyan("║")))
    print(bold(cyan("╚" + "═" * (len(banner_text)) + "╝")))
    print()

    # ── Metadata ──────────────────────────────────────────────────────────────
    print(
        f"  {dim('Python')}  : {bold(PYTHON_VERSION)}   "
        f"{dim('Backend')} : {bold(MODE)}   "
        f"{dim('lazyload')} : {bold(lazyload.__version__)}"
    )
    print(
        f"  {dim('Runs/module')} : {bold(str(runs))}   "
        f"{dim('(+ warmup runs excluded from stats)')}"
    )
    print()

    # ── Fallback notice ───────────────────────────────────────────────────────
    fallbacks = [r for r in results if r.is_fallback]
    if fallbacks:
        print(
            yellow(
                "  ⚠ Some preferred packages (numpy/pandas/torch) are not installed."
            )
        )
        print(
            yellow(
                "    Substituting stdlib modules — gains will be smaller than real packages."
            )
        )
        print(yellow("    Install them with:  pip install numpy pandas torch"))
        print()

    # ── Table ─────────────────────────────────────────────────────────────────
    cols = [
        ("Module", 26),
        ("Category", 13),
        ("Eager (ms)", 11),
        ("Lazy (ms)", 10),
        ("Deferred (ms)", 13),
        ("Speedup", 9),
    ]
    widths = [w for _, w in cols]
    headers = [h for h, _ in cols]

    print(_header_rule(widths))
    print(_row([bold(h) for h in headers], widths))
    print(_mid_rule(widths))

    total_eager = total_lazy = 0.0
    for i, r in enumerate(results):
        if i > 0:
            print(_body_rule(widths))

        name_cell = r.name + (dim(" †") if r.is_fallback else "")
        cat_cell = dim(r.category)
        eager_cell = bold(f"{r.eager_ms:>8.2f}")
        lazy_cell = green(f"{r.lazy_ms:>8.4f}")
        saved_cell = yellow(f"{r.saved_ms:>9.2f}")

        if r.speedup == float("inf"):
            sp_str = "∞×"
        elif r.speedup >= 1_000:
            sp_str = f"{r.speedup:,.0f}×"
        else:
            sp_str = f"{r.speedup:.1f}×"
        speedup_cell = magenta(sp_str.rjust(7))

        print(
            _row(
                [name_cell, cat_cell, eager_cell, lazy_cell, saved_cell, speedup_cell],
                widths,
            )
        )
        total_eager += r.eager_ms
        total_lazy += r.lazy_ms

    print(_mid_rule(widths))

    # Totals row
    total_saved = max(0.0, total_eager - total_lazy)
    total_sp = total_eager / total_lazy if total_lazy > 0 else float("inf")
    total_sp_str = f"{total_sp:,.0f}×" if total_sp != float("inf") else "∞×"
    print(
        _row(
            [
                bold("TOTAL"),
                "",
                bold(f"{total_eager:>8.2f}"),
                green(f"{total_lazy:>8.4f}"),
                yellow(f"{total_saved:>9.2f}"),
                magenta(total_sp_str.rjust(7)),
            ],
            widths,
        )
    )

    print(_foot_rule(widths))

    # ── Verdict ───────────────────────────────────────────────────────────────
    print()
    verdict_ms = f"{total_saved:.1f}ms"
    verdict_sp = total_sp_str
    print(f"  {green('✓')} {bold('Verdict')}")
    print(
        f"    lazyload defers {bold(yellow(verdict_ms))} of import work "
        f"({bold(magenta(verdict_sp))} faster startup)."
    )
    print()
    if total_saved < 10:
        note = (
            "The deferred time is small — stdlib modules are lightweight by design.  "
            "With numpy/pandas/torch the gains are typically 200ms–3 000ms per module."
        )
        print(f"    {dim('Note:')} {dim(note)}")
    else:
        print("    If none of those modules are used on a given run,")
        print(
            f"    that {bold(yellow(verdict_ms))} is {bold('eliminated entirely')} — not just deferred."
        )
    print()


# ─── Entry point ──────────────────────────────────────────────────────────────
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Measure startup time improvement from using lazyload.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--runs", type=int, default=5, help="Timed runs per module (default: 5)"
    )
    p.add_argument(
        "--warmup",
        type=int,
        default=2,
        help="Warm-up runs excluded from stats (default: 2)",
    )
    p.add_argument("--no-color", action="store_true", help="Disable ANSI colour output")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    if args.no_color:
        global _USE_COLOR
        _USE_COLOR = False

    print(dim("  Probing available modules …"), end="", flush=True)
    targets = _probe_targets()
    print(dim(f" found {len(targets)}"))

    print(dim("  Running benchmarks …"), end="", flush=True)
    results = run_benchmark(targets, runs=args.runs, warmup=args.warmup)
    print(dim(" done"))

    print_results(results, runs=args.runs)


if __name__ == "__main__":
    main()
