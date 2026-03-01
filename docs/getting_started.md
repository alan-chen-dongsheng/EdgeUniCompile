# EdgeUniCompile - Edge Universal Compile Framework

## Getting Started Guide

### What is EdgeUniCompile?

EdgeUniCompile is a comprehensive compiler framework designed specifically for edge AI devices. It addresses the challenges of running AI models on resource-constrained edge devices by providing:
- Efficient model conversion from ONNX to FlatBuffer format
- Mixed C++/Python compilation pass infrastructure for flexibility
- SRAM-aware tiling to optimize memory usage
- Integration with MLIR for advanced optimizations
- Target-specific code generation

### Project Overview

This is what I want to do:
1. Create a compiler framework for edge AI devices
2. Support ONNX to FlatBuffer conversion
3. Enable C++/Python mixed pass system
4. Optimize operations to fit SRAM size constraints
5. Integrate with MLIR for advanced optimizations

This is what I asked:
- Use C++17 and Python 3.13
- Build system: CMake + uv (package manager)
- Default compiler: Clang
- Support ONNX model to FlatBuffer conversion
- Enable C++/Python mixed passes
- Implement SRAM-aware tiling
- Integrate with MLIR

### How We Did It Now

We've established a complete project structure with:

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
├── examples/                     # Example projects
└── build/                       # Build directory (gitignored)
```

### Installation and Setup

#### Prerequisites

1. **Python 3.13+**: Make sure Python 3.13 or later is installed
2. **Clang Compiler**: Default compiler for the project
3. **CMake**: For building C++ components
4. **uv**: Python package manager (installed automatically)

#### Step-by-Step Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd EdgeUniCompile
   ```

2. **Create virtual environment and install dependencies**:
   ```bash
   uv venv
   uv pip install -e .
   ```

3. **Build C++ components**:
   ```bash
   mkdir build
   cd build
   cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=clang++
   make -j$(sysctl -n hw.ncpu)
   ```

4. **Run tests**:
   ```bash
   # Python tests
   uv run pytest tests/python/test_basic.py -v

   # C++ tests (not yet implemented)
   # cd build
   # ctest
   ```

### Usage Examples

#### Basic Usage

```python
import edgeunicompile as euc

# Create a compilation context
ctx = euc.Context()
ctx.sram_size = 16 * 1024 * 1024  # 16MB SRAM
ctx.opt_level = 3

# Create a simple graph
from edgeunicompile.ir import Graph, Node, Tensor

graph = Graph('test_graph')

# Add tensors
t1 = Tensor('input1', 'float32', (2, 3))
t2 = Tensor('input2', 'float32', (2, 3))
t3 = Tensor('output', 'float32', (2, 3))

graph.add_tensor(t1)
graph.add_tensor(t2)
graph.add_tensor(t3)
graph.add_input_tensor(t1)
graph.add_input_tensor(t2)
graph.add_output_tensor(t3)

# Add node
node = Node('add_node', 'Add')
node.add_input(t1)
node.add_input(t2)
node.add_output(t3)
graph.add_node(node)

# Create pass manager and run passes
from edgeunicompile.passes import PassManager

pass_manager = PassManager(ctx)
optimized_graph = pass_manager.run(graph)

# Compile to MLIR
from edgeunicompile.mlir import MLIRContext

mlir_ctx = MLIRContext(ctx)
mlir_module = mlir_ctx.compile(optimized_graph)

# Generate code
code = mlir_module.generate_code('cpu')
print(code)
```

#### ONNX Conversion

```python
import edgeunicompile as euc

# Convert ONNX model
converter = euc.ONNXConverter()
ctx = euc.Context()

# Load and convert ONNX model
try:
    graph = converter.convert('model.onnx', ctx)
    print(f"Successfully converted ONNX model to graph")
    print(f"Number of nodes: {len(graph.nodes)}")
    print(f"Number of tensors: {len(graph.tensors)}")
except Exception as e:
    print(f"Error converting ONNX model: {e}")
```

#### MLIR Installation

```python
import edgeunicompile as euc

# Check if MLIR is installed
if not euc.MLIRInstaller.is_installed():
    print("Installing MLIR...")
    success = euc.MLIRInstaller.install()
    if success:
        print(f"MLIR installed successfully (version: {euc.MLIRInstaller.get_version()})")
    else:
        print("MLIR installation failed")
```

### Build System Configuration

**CMakeLists.txt (Root):**
- C++17 standard
- Clang compiler configuration
- FetchContent for dependencies if not found
- Target definitions and installation

**pyproject.toml:**
- uv package manager configuration
- Python 3.13 dependencies
- Package metadata

**CMake Modules:**
- `FindFlatBuffers.cmake`: Finds or downloads FlatBuffers
- `FindMLIR.cmake`: Finds or provides MLIR integration

### Dependencies

**C++:**
- **FlatBuffers**: Automatic download if not found (v24.3.25)
- **MLIR**: Optional (will be installed via Python if needed)
- **GTest**: For testing (downloaded if not found)

**Python:**
- **numpy>=2.0**: Array operations
- **onnx>=1.16**: ONNX format support
- **flatbuffers>=24.0**: Serialization
- **click>=8.1**: CLI
- **rich>=13.7**: Rich output
- **pytest>=8.0**: Testing
- **pyyaml>=6.0**: Configuration

### Current Status

The project is now at a **foundational stage** with core infrastructure implemented:

✅ **Completed:**
- C++ core types and context management
- C++ IR (Graph, Node, Tensor) with tests
- Python package with core types and graph representation
- Basic CMake build system with dependency management
- Documentation structure
- Project README

⚠️ **Partially Completed:**
- Full pass system implementation
- MLIR integration is outlined but not fully implemented
- ONNX converter is stubbed but not fully functional
- FlatBuffer schema and converter need completion

### Future Roadmap

1. **Complete Pass System:** Implement all pass infrastructure
2. **ONNX Converter:** Complete ONNX to FlatBuffer/Python Graph
3. **MLIR Integration:** Full MLIR installation and usage
4. **Code Generation:** Instruction lowering from Graph IR
5. **Optimization Passes:** Constant folding, dead code elimination, tiling
6. **Target Support:** Add target architectures (ARM, RISC-V)
7. **Examples and Tests:** Complete example projects and test coverage

### Design Decisions

**Architecture Principles:**
- **Separation of concerns:** Clear layers between IR, passes, and lowering
- **Mixed language support:** C++ for performance-critical paths, Python for flexibility
- **Modular design:** Easy to add new passes, operations, and targets
- **FlatBuffers:** For compact serialization on edge devices
- **MLIR:** For advanced optimizations and code generation

### Key Files to Reference

- `README.md`: Project overview and quick start
- `include/edgeunicompile/core/types.h`: Core type system
- `include/edgeunicompile/ir/graph.h`: Graph IR representation
- `python/edgeunicompile/core/__init__.py`: Python types
- `python/edgeunicompile/mlir/installer.py`: MLIR installation

This framework provides a solid foundation for edge AI compilation and optimization.
