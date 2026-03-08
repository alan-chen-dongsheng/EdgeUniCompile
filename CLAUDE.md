```
# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.
```

## EdgeUniCompile - Edge Universal Compile Framework

A comprehensive compiler framework designed specifically for edge AI devices. It supports:

- ONNX to FlatBuffer conversion
- Mixed C++/Python pass system
- SRAM-aware tiling
- MLIR integration
- Target-specific code generation

## rules

1. make sure you can build the project success if you do code changes.
2. For all CMake projects, use git submodules as the default package management method. Do not use FetchContent.

## Project Architecture

```
EdgeUniCompile/
├── include/edgeunicompile/       # Public C++ headers
│   ├── core/                     # Core types and context
│   ├── ir/                       # Intermediate representation
│   ├── passes/                   # Pass infrastructure
│   ├── flatbuf/                  # FlatBuffer support
│   ├── lowering/                 # Lowering pipelines
│   ├── tiling/                   # Tiling and memory planning
│   └── mlir/                     # MLIR integration
├── src/                          # C++ implementation
├── python/edgeunicompile/        # Python package
│   ├── core/                     # Python types
│   ├── ir/                       # Python IR representation
│   ├── onnx/                     # ONNX conversion
│   ├── flatbuf/                  # FlatBuffer support
│   ├── passes/                   # Pass infrastructure
│   ├── mlir/                     # MLIR integration
│   └── cli/                      # Command-line interface
├── tests/                        # Tests
├── docs/                         # Documentation
└── examples/                     # Example projects
```

## Getting Started

### Build System

**Dependencies:**

- C++17
- Python 3.13
- Clang (default compiler)
- CMake + uv (package manager)

### Installation and Setup

```bash
# Create virtual environment and install Python dependencies
uv venv
uv pip install -e .

# Build C++ components
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=clang++
make -j$(sysctl -n hw.ncpu)
```

### Running Tests

```bash
# Python tests
uv run pytest tests/python/test_basic.py -v

# C++ tests (not yet implemented)
# cd build && ctest
```

### Example Usage

```bash
# Run minimal example
uv run python examples/minimal_example.py

# Run simple example
uv run python examples/simple_example.py
```

### Common Development Tasks

```bash
# Install package in development mode
uv pip install -e .

# Run Python tests with coverage
uv run pytest tests/python/test_basic.py -v --tb=short

# Format Python code (requires pre-commit)
uv run python -m pip install pre-commit
uv run pre-commit install
uv run pre-commit run --all-files

# Clean build artifacts
rm -rf build/
rm -rf .pytest_cache/
rm -rf __pycache__/
```

## Key Components

## Key Components

### C++ Core

- **Core Types:** `/include/edgeunicompile/core/types.h` - DataType, Shape, OpType, AttributeValue
- **Context:** `/include/edgeunicompile/core/context.h` - CompileContext, Context
- **Graph IR:** `/include/edgeunicompile/ir/` - Graph, Node, Tensor
- **Passes:** `/include/edgeunicompile/passes/` - PassBase, PassManager, ConstantFoldingPass
- **FlatBuffer:** `/include/edgeunicompile/flatbuf/` - FlatBufferConverter, edgeunicompile.fbs schema

### Python Package

- **Core Types:** `/python/edgeunicompile/core/__init__.py` - Context, Status, Shape
- **Graph IR:** `/python/edgeunicompile/ir/__init__.py` - Graph, Node, Tensor
- **ONNX Conversion:** `/python/edgeunicompile/onnx/__init__.py` - ONNXConverter
- **Pass System:** `/python/edgeunicompile/passes/__init__.py` - PassBase, PassManager
- **Tiling Pass:** `/python/edgeunicompile/passes/tiling_pass.py` - TilingPass (slice-compute-concat)
- **FlatBuffer Support:** `/python/edgeunicompile/flatbuf/__init__.py` - FlatBufferBuilder, schema-based serialization
- **MLIR Integration:** `/python/edgeunicompile/mlir/__init__.py` - MLIRInstaller, MLIRContext

### Build Configuration

- **Root CMakeLists:** `/CMakeLists.txt` - Main build configuration
- **Src CMakeLists:** `/src/CMakeLists.txt` - Library build with FlatBuffer code generation
- **Python Package:** `/pyproject.toml` - uv package manager configuration
- **CMake Modules:** `/cmake/` - FindFlatBuffers.cmake, FindMLIR.cmake

### FlatBuffer Schema

The FlatBuffer schema (`/include/edgeunicompile/flatbuf/edgeunicompile.fbs`) defines the shared IR:
- `DataType` enum - Tensor data types
- `OpType` enum - Operation types (Conv2D, Slice, Concat, etc.)
- `MemoryLocation` enum - DRAM, SRAM, L2Cache, Register
- `Graph`, `Node`, `Tensor`, `Shape`, `Attribute` tables

To regenerate FlatBuffer code:
```bash
# C++ (automatic via CMake if flatc is found)
flatc -c --gen-mutable include/edgeunicompile/flatbuf/edgeunicompile.fbs

# Python
flatc -p --gen-mutable include/edgeunicompile/flatbuf/edgeunicompile.fbs
```

## Dependencies

### C++

- **FlatBuffers:** Git submodule (v24.3.25) in `third_party/flatbuffers`
- **MLIR:** Optional (will be installed via Python if needed)
- **GTest:** Git submodule (v1.15.0) in `third_party/googletest`
- **pybind11:** Git submodule (v2.11.1) in `third_party/pybind11` (optional)

**Initialize submodules:**
```bash
git submodule update --init --recursive
```

### Python

- **numpy>=2.0**: Array operations
- **onnx>=1.16**: ONNX format support
- **flatbuffers>=24.0**: Serialization
- **click>=8.1**: CLI
- **rich>=13.7**: Rich output
- **pytest>=8.0**: Testing
- **pyyaml>=6.0**: Configuration

## Current Status

The project has a **solid foundation** with core infrastructure implemented.

### Completed:

- C++ core types and context management
- C++ IR (Graph, Node, Tensor) with tests
- **C++ Pass System** - PassBase, PassManager, ConstantFoldingPass
- **FlatBuffer Schema** - edgeunicompile.fbs for shared C++/Python IR
- **FlatBuffer Converter** - C++ and Python serialization/deserialization
- **Python Tiling Pass** - Slice-compute-concat pattern for SRAM-aware compilation
- Python package with core types and graph representation
- Basic CMake build system with dependency management
- Documentation structure
- Project README

### In Progress:

- MLIR integration is outlined but not fully implemented
- ONNX converter is stubbed but not fully functional
- Additional C++ passes (dead code elimination, etc.)

## Design Decisions

**Architecture Principles:**

- **Separation of concerns:** Clear layers between IR, passes, and lowering
- **Mixed language support:** C++ for performance-critical paths, Python for flexibility
- **Modular design:** Easy to add new passes, operations, and targets
- **FlatBuffers:** For compact serialization on edge devices
- **MLIR:** For advanced optimizations and code generation

**Edge Device Constraints:**

- **SRAM-aware tiling:** Optimize operations to fit within SRAM size
- **Memory planning:** Efficient use of limited memory resources
- **Compact serialization:** FlatBuffers for small model sizes
