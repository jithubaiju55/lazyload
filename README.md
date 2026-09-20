# lazyload

> Eliminate Python startup latency caused by heavy, eager imports.

[![PyPI version](https://badge.fury.io/py/lazyload-py.svg)](https://pypi.org/project/lazyload-py/)
[![Python Versions](https://img.shields.io/pypi/pyversions/lazyload-py.svg)](https://pypi.org/project/lazyload-py/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Coverage](https://img.shields.io/badge/coverage-93%25-brightgreen.svg)](https://github.com/jithubaiju55/lazyload)

`lazyload` defers Python module loading until the exact moment an attribute is accessed for the first time, delivering instant application startup with zero configuration.

---

## The Problem

Standard Python imports execute eagerly at module load time. If your CLI tool or web service imports PyTorch, Pandas, or NumPy at the top of the file, Python parses C-extensions, allocates memory, and runs module initialization code immediately—even when executing simple flags like `--help` or handling fast paths that never use those dependencies. Loading heavy libraries eagerly turns what should be a sub-10ms CLI command into a frustating 4-second wait.

```python
# Standard eager imports pay 4,000ms startup cost on EVERY execution
import torch
import pandas as pd
import numpy as np

def main():
    if "--help" in sys.argv:
        print_help()  # Loaded PyTorch just to print text!
        return
    model = torch.load("model.pt")
```

---

## The Solution

`lazyload` replaces eager module loading with deferred proxies that register in `sys.modules` instantly. One line of code defers all import work until an attribute is actually accessed during execution. If an execution branch never uses the imported library, the import overhead is eliminated completely.

```python
# One line changes everything — <15ms startup time
import lazyload

torch = lazyload.lazy("torch")
pd = lazyload.lazy("pandas")
np = lazyload.lazy("numpy")
```

Running `--help` now returns in 12 milliseconds.

---

## Features

- **Universal Python Support**: Works across Python 3.10, 3.11, 3.12, 3.13, 3.14, and 3.15+.
- **Zero External Dependencies**: Lightweight pure-Python implementation.
- **Three Import Styles**: Support for explicit function deferral (`lazy`), context blocks (`lazy_imports`), and function decorators (`@lazy_module`).
- **Native 3.15 Fast Path**: Automatically leverages Python 3.15+ native PEP 810 lazy module capabilities for interpreter-level execution.
- **Compatible Shim for Older Versions**: Provides an identical deferred proxy pattern for Python 3.10–3.14.
- **Drop-in Integration**: No codebase restructuring or module layout changes required.

---

## Installation

```bash
pip install lazyload-py
```

---

## Quick Start

### 1. Single Module Deferral (`lazy`)

Defer a single heavy library by passing its module name as a string.

```python
import lazyload

# Registers a deferred proxy instantly (<0.01ms)
torch = lazyload.lazy("torch")

# PyTorch is loaded here, on first attribute access
tensor = torch.tensor([1.0, 2.0, 3.0])
```

### 2. Grouped Import Block (`lazy_imports`)

Defer multiple standard import statements inside a clean context manager block without altering your import syntax.

```python
import lazyload

# All imports declared inside the block are deferred automatically
with lazyload.lazy_imports():
    import numpy as np
    import pandas as pd
    import scipy

# Execution continues instantly; modules load on first attribute lookup
df = pd.DataFrame({"data": [1, 2, 3]})
```

### 3. Function Decorator (`lazy_module`)

Decorate entry points or CLI handlers to defer all top-level imports inside the function until the function is called for the first time.

```python
import lazyload

@lazyload.lazy_module
def run_training_pipeline():
    # Imports inside the function are deferred until execution
    import torch
    import torchvision

    print("Pipeline started.")
```

---

## How It Works

On Python 3.15 and above, `lazyload` delegates directly to the Python interpreter's native PEP 810 lazy module mechanism, leveraging internal engine hooks to eliminate proxy wrapping overhead and allowing the interpreter itself to handle deferred module evaluation.

On Python 3.10 through 3.14, `lazyload` installs a lightweight proxy object into `sys.modules` under the target module name. The proxy transparently intercepts attribute lookups, performs the real import on first access, replaces itself in `sys.modules` with the real module, and updates its internal attribute dictionary so subsequent lookups carry zero ongoing overhead.

---

## Benchmarks

Measured on a CLI entry point importing PyTorch, Pandas, and NumPy on Python 3.10:

| Import Approach | Startup Time | Deferred Overhead | Speedup |
| :--- | :--- | :--- | :--- |
| **Eager Standard (`import torch, pandas, numpy`)** | `4,112 ms` | `0 ms` | `1.0×` |
| **`lazyload` Proxy (`lazyload.lazy(...)`)** | **`12 ms`** | **`4,100 ms`** | **`342.6×`** |

*Run the benchmark suite locally using `python benchmarks/bench_startup.py` and `python benchmarks/bench_overhead.py`.*

---

## Contributing

Contributions are welcome! Please review [CONTRIBUTING.md](CONTRIBUTING.md) for developer setup, code guidelines, and testing procedures.

---

## License

`lazyload` is licensed under the [MIT License](LICENSE).
