# AGENTS.md - Agentic Coding Guidelines for EdgeUniCompile

This file provides guidance for agentic coding agents working in the EdgeUniCompile repository.

## Project Overview

EdgeUniCompile is an edge AI compiler framework with:
- **C++ core** for performance-critical paths
- **Python package** for flexibility and ease of use
- **Mixed C++/Python pass system** for optimizations
- **FlatBuffers** for compact serialization on edge devices
- **MLIR integration** for advanced code generation

---

## Build, Lint, and Test Commands

### Python Environment Setup

```bash
# Create virtual environment and install dependencies
uv venv
uv pip install -e .

# Or with pip
python3.13 -m venv .venv
source .venv/bin/activate
pip install -e .
```

### Running Python Tests

```bash
# Run all Python tests
uv run pytest tests/python/ -v

# Run a single test file
uv run pytest tests/python/test_basic.py -v

# Run a single test function
uv run pytest tests/python/test_basic.py::test_graph_creation -v

# Run with verbose output and short traceback
uv run pytest tests/python/ -v --tb=short

# Run with coverage (if coverage is installed)
uv run pytest tests/python/ --cov=edgeunicompile --cov-report=term-missing
```

### C++ Build

```bash
# Initialize git submodules (first time only)
git submodule update --init --recursive

# Configure and build (from project root)
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=clang++
make -j$(sysctl -n hw.ncpu)

# Build specific target
make EdgeUniCompile -j$(sysctl -n hw.ncpu)
```

### Running C++ Tests

```bash
cd build
ctest --output-on-failure

# Or run a specific test
./bin/edgeunicompile_tests
```

### Code Quality

```bash
# Format Python code (if pre-commit is set up)
uv run pre-commit install
uv run pre-commit run --all-files

# Format manually with black (if installed)
uv run black python/edgeunicompile/

# Lint with ruff (if installed)
uv run ruff check python/edgeunicompile/

# Clean build artifacts
rm -rf build/
rm -rf .pytest_cache/
rm -rf __pycache__/
rm -rf python/edgeunicompile/__pycache__/
```

---

## Code Style Guidelines

### Python Style

**Imports**
- Use absolute imports: `from edgeunicompile.core import Shape`
- Group imports: stdlib first, then third-party, then local
- Sort imports alphabetically within groups
- Use `from typing import List, Optional, Dict` for type hints

**Formatting**
- Maximum line length: 100 characters
- Use 4 spaces for indentation (no tabs)
- Use Black formatting if available
- Use blank lines sparingly to group related code (max 2 consecutive)

**Type Hints**
- Always use type hints for function parameters and return types
- Use `Optional[X]` instead of `X | None` for Python 3.13 compatibility
- Use `List[X]`, `Dict[K, V]` from typing module

**Naming Conventions**
- Classes: `PascalCase` (e.g., `class GraphBuilder`)
- Functions/methods: `snake_case` (e.g., `def get_tensor`)
- Variables: `snake_case` (e.g., `node_name`)
- Constants: `UPPER_SNAKE_CASE` (e.g., `MAX_BUFFER_SIZE`)
- Private methods/variables: prefix with underscore (e.g., `_internal_map`)

**Dataclasses**
- Use `@dataclass` for simple data containers
- Use `@dataclass` with `frozen=True` for immutable types
- Use `field(default_factory=list)` for mutable defaults

**Error Handling**
- Use specific exception types: `ValueError`, `TypeError`, `RuntimeError`
- Include descriptive error messages: `raise ValueError(f"Invalid shape: {shape}")`
- Use `Status` object from `edgeunicompile.core` for operations that may fail
- Check validity early: validate inputs at function entry

**Docstrings**
- Use triple quotes `"""` for docstrings
- Follow Google style: `"""Do something.\n\nArgs:\n    x: Description.\n\nReturns:\n    Description."""
- Keep brief: one-line for simple functions, multi-line only when needed

### C++ Style

**General**
- C++17 standard required
- Use Clang compiler with libc++ (`-stdlib=libc++`)
- Enable strict warnings: `-Wall -Wextra -Wpedantic -Werror`
- Disable RTTI for MLIR compatibility: `-fno-rtti`

**Naming**
- Classes: `PascalCase` (e.g., `class CompileContext`)
- Functions: `snake_case` (e.g., `void add_node()`)
- Member variables: `snake_case_` with trailing underscore (e.g., `std::string name_`)
- Constants: `kPascalCase` (e.g., `const int kMaxSize = 1024`)

**Headers**
- Use `#pragma once` instead of include guards
- Order includes: related header, then project headers, then external libs, then stdlib
- Use forward declarations when possible to reduce compile times

**Memory Management**
- Prefer smart pointers (`std::unique_ptr`, `std::shared_ptr`) over raw pointers
- Use `std::vector` for dynamic arrays
- Avoid manual memory allocation

---

## Project-Specific Conventions

### Pass System

- All passes should inherit from `PassBase` (C++) or `PassBase` (Python)
- Passes must implement `run()` method
- Use `PassManager` to orchestrate pass execution
- Document pass requirements and guarantees

### IR (Intermediate Representation)

- `Graph` contains `Node` and `Tensor` objects
- Use `Node.op_type` from `OpType` enum
- Use `Tensor.dtype` from `DataType` enum
- Validate graph with `graph.is_valid()` before processing

### FlatBuffers

- Schema defined in `include/edgeunicompile/flatbuf/edgeunicompile.fbs`
- Regenerate code when schema changes:
  ```bash
  flatc -c --gen-mutable include/edgeunicompile/flatbuf/edgeunicompile.fbs
  flatc -p --gen-mutable include/edgeunicompile/flatbuf/edgeunicompile.fbs
  ```

### Tiling Pass

- Implement slice-compute-concat pattern for SRAM-aware compilation
- Use `TilingOptions` to configure tile sizes
- Validate tiles fit in SRAM before applying

---

## Common Development Tasks

1. **Adding a new operation**: Add to `OpType` enum in core, implement in passes
2. **Adding a new pass**: Inherit from `PassBase`, implement `run()`, register in `PassManager`
3. **Adding a new data type**: Add to `DataType` enum and `get_data_type_size()`
4. **Modifying FlatBuffer schema**: Update `.fbs` file, regenerate code, update converters

---

## Testing Guidelines

- Place Python tests in `tests/python/` as `test_*.py`
- Use pytest fixtures for common setup
- Test both success and failure cases
- Include edge cases (empty inputs, max values, etc.)
- Name test functions: `test_<what>_<expected_behavior>` (e.g., `test_graph_add_node_raises_on_duplicate`)

---

## File Locations

- **Python package**: `python/edgeunicompile/`
- **C++ headers**: `include/edgeunicompile/`
- **C++ source**: `src/`
- **Python tests**: `tests/python/`
- **C++ tests**: `tests/cpp/`
- **Examples**: `examples/`
