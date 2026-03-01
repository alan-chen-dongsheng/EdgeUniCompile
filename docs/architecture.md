# EdgeUniCompile - Edge Universal Compile Framework

## 1. What I Want to Do

I want to create a comprehensive compiler framework designed specifically for edge AI devices, addressing the unique challenges of running AI models on resource-constrained edge devices. The goal is to build a flexible and efficient compiler that can:

- Convert ONNX models to a more compact and efficient format (FlatBuffers)
- Provide a mixed C++/Python pass system for model optimizations
- Optimize operations to fit within SRAM size constraints (a critical limitation for edge devices)
- Lower computation graphs to target-specific instructions
- Integrate with MLIR (Multi-Level Intermediate Representation) for advanced optimizations
- Enable both C++ and Python compilation passes

## 2. What I Asked

**Technical Requirements:**
- Use C++17
- Use Python 3.13
- Build system: CMake + uv (package manager)
- Default compiler: Clang

**Key Features:**
1. ONNX model to FlatBuffer conversion (Python)
2. FlatBuffer model import into C++ objects
3. Mixed C++/Python compilation pass system
4. Lowering DAG networks to instructions
5. Tiling operations to fit SRAM size constraints
6. MLIR integration with Python package manager support
7. Using MLIR features for various compilation tasks

## 3. How We Did It Now

### Project Structure Created

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
├── third_party/                  # Third-party dependencies
├── CMakeLists.txt               # Root CMake configuration
└── pyproject.toml              # Python package definition
```

### Core Components Implemented

#### C++ Core (C++17)

1. **Types and Context Management**
   - `DataType`: Supported data types (float32, float16, int32, etc.)
   - `Shape`: Tensor shape representation
   - `OpType`: Operation types (conv2d, relu, matmul, etc.)
   - `CompileContext`: Compiler configuration and state management

2. **Intermediate Representation (IR)**
   - `Tensor`: Tensor representation with data and metadata
   - `Node`: Operation with inputs, outputs, and attributes
   - `Graph`: Computation graph managing nodes and tensors

3. **Pass System Infrastructure**
   - `PassBase`: Base class for all passes
   - `PassManager`: Manages pass execution
   - Support for both C++ and Python passes

4. **FlatBuffer Support**
   - Schema definition for model serialization
   - Importer and exporter functionality

5. **Tiling and Memory Planning**
   - `TilingPass`: SRAM-aware operation tiling
   - `MemoryModel`: Memory usage analysis

6. **MLIR Integration**
   - MLIR context and integration layer
   - Custom dialect (EdgeOps)
   - Lowering pipelines

#### Python Package (Python 3.13)

1. **Core Types**
   - Dataclass-based implementations of C++ core types
   - `Context` for configuration management

2. **Graph IR**
   - Python versions of `Graph`, `Node`, and `Tensor`
   - Serialization support

3. **ONNX Conversion**
   - `ONNXConverter`: Converts ONNX models to EdgeUniCompile Graph
   - Support for standard ONNX operations

4. **Pass System**
   - `PassBase` and `PassManager` in Python
   - Compatibility with C++ passes

5. **MLIR Integration**
   - `MLIRInstaller`: Installs MLIR dependencies via Python
   - `MLIRContext`: Wraps MLIR API for Python
   - `MLIRCompiler`: Compiles Graph IR to MLIR

6. **CLI Interface**
   - Command-line tool for model conversion and compilation
   - Supports various input and output formats

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
- `EdgeUniCompileConfig.cmake.in`: Package configuration

### Dependencies Handled

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

## 4. Everything Important to Know

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

### Usage Examples

**Python API:**
```python
import edgeunicompile as euc

# Load ONNX model
model_path = "examples/simple_conv/model.onnx"
graph = euc.ONNXConverter.convert(model_path)

# Create context with SRAM size (e.g., 32MB)
ctx = euc.Context(sram_size=32 * 1024 * 1024)

# Create pass manager and run passes
pass_manager = euc.PassManager(ctx)
optimized_graph = pass_manager.run(graph)

# Compile to MLIR
mlir_ctx = euc.MLIRContext(ctx)
mlir_module = mlir_ctx.compile(optimized_graph)

# Generate code
code = mlir_ctx.generate_code(mlir_module)
print(code)
```

**C++ API:**
```cpp
#include <edgeunicompile/core/context.h>
#include <edgeunicompile/ir/graph.h>
#include <edgeunicompile/flatbuf/importer.h>

using namespace edgeunic;

int main() {
    auto ctx = Context::Create();
    ctx->SetSramSize(32 * 1024 * 1024);
    ctx->SetOptLevel(3);

    auto graph = FlatBufferImporter::Import("model.fb", ctx);

    // Run passes
    PassManager manager(ctx);
    auto optimized_graph = manager.Run(graph);

    // Lower to instructions
    LoweringPipeline pipeline(ctx);
    auto instructions = pipeline.Lower(optimized_graph);

    return 0;
}
```

### Build Instructions

```bash
# Setup virtual environment
uv venv
uv pip install -r requirements.txt

# Build C++
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=clang++
make -j$(sysctl -n hw.ncpu)

# Install Python package
uv pip install -e .

# Run tests
pytest tests/python/
```

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

### Challenges

- **Edge device constraints:** Limited memory (SRAM) and computational power
- **Mixed language complexity:** Handling Python-C++ interaction
- **SRAM-aware tiling:** Finding optimal tile sizes
- **MLIR installation:** Large dependency via Python package manager

### Key Files to Reference

- `README.md`: Project overview and quick start
- `include/edgeunicompile/core/types.h`: Core type system
- `include/edgeunicompile/ir/graph.h`: Graph IR representation
- `python/edgeunicompile/core/__init__.py`: Python types
- `python/edgeunicompile/mlir/installer.py`: MLIR installation

This framework provides a solid foundation for edge AI compilation and optimization.
