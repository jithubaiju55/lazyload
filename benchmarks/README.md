# lazyload Benchmarks

This directory contains standalone benchmark scripts designed to evaluate the performance characteristics of `lazyload`.

---

## Benchmark Scripts

### 1. `bench_startup.py` — Startup Time Deferral Benchmark

Measures how much time `lazyload` defers at application startup by substituting eager imports (`import numpy`, `import pandas`, `import torch`) with deferred proxy registrations (`lazyload.lazy(...)`).

#### What it measures
* **Eager Import Time (ms)**: Time taken to load target modules into `sys.modules` from cold disk cache.
* **Lazy Proxy Time (ms)**: Time taken to register a lightweight `lazyload` proxy.
* **Deferred Time (ms)**: The import startup overhead saved and deferred to first actual usage.
* **Speedup Factor**: Relative startup acceleration.

#### Running the benchmark
```bash
python benchmarks/bench_startup.py
```

##### Command-line Options
* `--runs N`: Number of timed measurement runs per module (default: `5`).
* `--warmup N`: Number of warm-up iterations excluded from statistics (default: `2`).
* `--no-color`: Disable ANSI colored terminal output.

---

### 2. `bench_overhead.py` — Runtime Proxy Overhead Benchmark

Measures the microsecond and nanosecond runtime overhead introduced by `lazyload`'s proxy mechanism across three core scenarios.

#### What it measures
1. **Proxy Creation Overhead**: Time to instantiate a deferred proxy object vs performing an eager import.
2. **First Attribute Access (Reification Overhead)**: Time taken for the first attribute access on a proxy (which triggers the real module import and proxy reification) vs eager import + access.
3. **Subsequent Attribute Access (Post-Reification Overhead)**: Time taken for attribute lookups on an already reified proxy vs direct attribute access on a standard module.

#### Running the benchmark
```bash
python benchmarks/bench_overhead.py
```

##### Command-line Options
* `--iterations N`: Number of iterations for each scenario (default: `1000`).
* `--no-color`: Disable ANSI colored terminal output.

---

## Architecture & Modes

Both benchmark scripts automatically detect the execution environment and report metrics based on the active backend mode:

* **Native Mode (PEP 810)**: Active on Python 3.15+. Uses interpreter-level native lazy imports with zero runtime proxy wrapping overhead.
* **Shim Mode**: Active on Python 3.10–3.14. Uses the `_DeferredProxy` shim to defer import execution until first attribute access.

---

## Requirements

The benchmark scripts are entirely standalone and require **no third-party dependencies** beyond Python standard library and `lazyload`. If heavy third-party packages (`numpy`, `pandas`, `torch`) are installed, `bench_startup.py` will automatically test them; otherwise, stdlib fallback modules are measured cleanly.
