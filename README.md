# EdgeUniCompile - Edge Universal Compile Framework

A comprehensive compiler framework designed specifically for edge AI devices, addressing the challenges of running AI models on resource-constrained edge devices.

## Features

- **ONNX to FlatBuffer Conversion**: Efficient model conversion from ONNX to FlatBuffer format using Python
- **Mixed C++/Python Pass System**: Flexible pass infrastructure supporting both C++ and Python passes
- **SRAM-Aware Tiling**: Optimizes operations to fit SRAM size constraints
- **DAG to Instructions Lowering**: Converts computation graphs to target instructions
- **MLIR Integration**: Advanced optimizations using MLIR framework
- **Tiling and Memory Planning**: Memory usage optimization for edge devices

## Technical Requirements

- **C++**: C++17
- **Python**: Python 3.13
- **Build System**: CMake + uv (package manager)
- **Default Compiler**: Clang

## Project Structure

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
│   ├── cpp/                      # C++ tests
│   └── python/                   # Python tests
├── docs/                         # Documentation
├── examples/                     # Example projects
├── third_party/                  # Third-party dependencies
├── CMakeLists.txt               # Root CMake configuration
└── pyproject.toml              # Python package definition
```

## Quick Start

### 1. Clone the Repository

```bash
git clone <repository-url>
cd EdgeUniCompile
```

### 2. Create Virtual Environment and Install Dependencies

```bash
uv venv
uv pip install -r requirements.txt
```

### 3. Build C++ Components

```bash
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=clang++
make -j$(nproc)
```

### 4. Install Python Package

```bash
uv pip install -e .
```

### 5. Run Tests

```bash
# Python tests
pytest tests/python/

# C++ tests
cd build
ctest
```

## Usage Example

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

## Key Components

### ONNX Converter

The ONNX converter transforms ONNX models to EdgeUniCompile's internal graph representation. It supports:

- Convolution (Conv2D)
- Pooling (MaxPool2D, AveragePool2D)
- Activations (ReLU, Sigmoid, Tanh)
- MatMul
- Element-wise operations (Add, Subtract, Multiply, Divide)
- Reshape and Transpose

### Pass System

The pass system supports both C++ and Python passes. Passes can modify the graph for optimization:

- Constant folding
- Dead code elimination
- Tiling
- Memory planning
- Optimization passes

### Tiling Pass

The tiling pass optimizes operations to fit SRAM size constraints. It:

1. Analyzes operation memory requirements
2. Determines optimal tile sizes
3. Generates tiled operations
4. Plans memory usage

### MLIR Integration

The MLIR integration provides advanced optimizations:

- Bufferization
- Loop optimizations
- Vectorization
- Target-specific lowering

## Building from Source

### Requirements

- CMake 3.20+
- Clang 16+
- Python 3.13+
- uv package manager

### Build Options

```bash
cmake .. -DCMAKE_BUILD_TYPE=Release \
         -DCMAKE_CXX_COMPILER=clang++ \
         -DEUC_BUILD_TESTS=ON \
         -DEUC_ENABLE_MLIR=ON \
         -DEUC_ENABLE_TILING=ON
```

## Contributing

Contributions are welcome! Please see CONTRIBUTING.md for details.

## License

MIT License. See LICENSE for details.
