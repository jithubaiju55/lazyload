# Contributing to lazyload

Thank you for your interest in contributing to **lazyload**! 

Whether you're fixing a typo in documentation, reporting a bug, optimizing performance, or implementing a new feature, your contribution is greatly appreciated. We strive to make this project welcoming and accessible to everyone—including first-time open-source contributors.

---

## Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [First-Time Contributors](#first-time-contributors)
3. [Development Setup](#development-setup)
4. [Running Tests](#running-tests)
5. [Linting & Type Checking](#linting--type-checking)
6. [Running Benchmarks](#running-benchmarks)
7. [Submitting a Pull Request](#submitting-a-pull-request)
8. [Getting Help](#getting-help)

---

## Code of Conduct

This project adheres to the [Contributor Covenant](https://www.contributor-covenant.org/) Code of Conduct. By participating in this project, you agree to maintain a respectful, supportive, and welcoming environment for everyone.

---

## First-Time Contributors

If this is your first time contributing to an open-source Python project:
- Don't worry if you get stuck or make a mistake! We are happy to help answer questions and guide you through the process.
- Check out issues labeled [`good first issue`](https://github.com/jithubaiju55/lazyload/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22) for beginner-friendly tasks.

---

## Development Setup

### 1. Fork & Clone

1. Click the **Fork** button at the top-right of the [`lazyload` GitHub repository](https://github.com/jithubaiju55/lazyload).
2. Clone your fork to your local machine:
   ```bash
   git clone https://github.com/jithubaiju55/lazyload.git
   cd lazyload
   ```

### 2. Create a Virtual Environment

Create and activate an isolated Python 3.10+ virtual environment:

**Linux / macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install in Editable Mode

Install `lazyload` in editable development mode along with all developer tools (`pytest`, `pytest-cov`, `ruff`, `mypy`):

```bash
pip install --upgrade pip
pip install -e ".[dev]"
```

---

## Running Tests

We use **[pytest](https://docs.pytest.org/)** for testing and **pytest-cov** for coverage analysis.

Run the entire test suite:
```bash
pytest
```

Run tests with line-by-line coverage output:
```bash
pytest --cov=lazyload --cov-report=term-missing
```

Run a specific test file:
```bash
pytest tests/test_api.py
```

---

## Linting & Type Checking

We maintain strict code quality standards using **Ruff** and **Mypy**.

### 1. Code Formatting & Linting (Ruff)

Check for linting errors across the codebase:
```bash
ruff check .
```

Automatically fix format and lint issues:
```bash
ruff format .
ruff check --fix .
```

### 2. Static Type Checking (Mypy)

Verify type annotations with Mypy strict mode:
```bash
mypy lazyload
```

---

## Running Benchmarks

`lazyload` includes two standalone benchmark scripts in the `benchmarks/` folder:

1. **Startup Deferral Benchmark**:
   ```bash
   python benchmarks/bench_startup.py
   ```
2. **Runtime Overhead Benchmark**:
   ```bash
   python benchmarks/bench_overhead.py
   ```

---

## Submitting a Pull Request

1. **Create a topic branch** off `main`:
   ```bash
   git checkout -b feat/my-new-feature
   ```
2. **Make your code changes** following the existing code style and adding tests for any new functionality.
3. **Verify all checks pass**:
   ```bash
   pytest
   ruff check .
   ruff format --check .
   mypy lazyload
   ```
4. **Commit your changes** with descriptive commit messages:
   ```bash
   git add .
   git commit -m "feat: add support for custom deferred proxy repr"
   ```
5. **Push your branch** to your fork:
   ```bash
   git push origin feat/my-new-feature
   ```
6. **Open a Pull Request** on GitHub against the `main` branch. Provide a concise summary of what your PR changes and why.

---

## Getting Help

If you run into any questions, feel free to open a [GitHub Issue](https://github.com/jithubaiju55/lazyload/issues) or comment directly on your active Pull Request. We're excited to collaborate with you!
