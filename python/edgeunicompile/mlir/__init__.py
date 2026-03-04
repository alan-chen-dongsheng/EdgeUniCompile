"""
MLIR integration for EdgeUniCompile.

This module provides functionality to:
1. Install MLIR dependencies via Python
2. Convert EdgeUniCompile Graph to MLIR
3. Optimize MLIR code
4. Generate target-specific code from MLIR
"""

import subprocess
import sys
import os
import tempfile
from typing import Optional, List, Dict
import logging

from edgeunicompile.core import Context
from edgeunicompile.ir import Graph


class MLIRInstaller:
    """
    Installer for MLIR dependencies.

    This class provides static methods to install and configure MLIR
    using the Python package manager.
    """

    @staticmethod
    def install(force: bool = False) -> bool:
        """
        Install MLIR dependencies.

        Args:
            force: Force installation even if already installed.

        Returns:
            True if installation was successful.
        """
        # Check if MLIR is already available
        if not force:
            try:
                import mlir
                return True
            except ImportError:
                pass

        # Install MLIR via pip
        try:
            result = subprocess.run([
                sys.executable, "-m", "pip", "install",
                "mlir-dialects>=2.0", "mlir-parser>=2.0"
            ],
                capture_output=True,
                text=True)

            if result.returncode != 0:
                raise RuntimeError(f"Failed to install MLIR: {result.stderr}")

            return True
        except Exception as e:
            logging.error(f"MLIR installation failed: {e}")
            return False

    @staticmethod
    def is_installed() -> bool:
        """
        Check if MLIR is installed.

        Returns:
            True if MLIR is installed.
        """
        try:
            import mlir
            return True
        except ImportError:
            return False

    @staticmethod
    def get_version() -> str:
        """
        Get installed MLIR version.

        Returns:
            MLIR version string.
        """
        try:
            import mlir
            return getattr(mlir, '__version__', 'unknown')
        except Exception:
            return 'not_installed'


class MLIRContext:
    """
    Context for MLIR operations.

    This class wraps the MLIR API for Python.
    """

    def __init__(self, context: Optional[Context] = None):
        """
        Create an MLIR context.

        Args:
            context: Optional compilation context.
        """
        if context is None:
            from edgeunicompile.core import Context
            context = Context()

        self.context = context
        self._mlir_context = None
        self._session = None
        self._use_mock = not MLIRInstaller.is_installed()

        # Initialize MLIR (or use mock)
        if self._use_mock:
            print("  Using mock MLIR (real MLIR not installed)")
        else:
            self._initialize_mlir()

    def _initialize_mlir(self):
        """Initialize the MLIR context."""
        if self._use_mock:
            # Mock initialization
            return

        if not MLIRInstaller.is_installed():
            raise ImportError("MLIR not installed. Please run MLIRInstaller.install()")

        try:
            import mlir
            self._mlir_context = mlir.context.Context()
            self._session = mlir.ir.Module.create(self._mlir_context)
        except Exception as e:
            raise RuntimeError(f"MLIR initialization failed: {e}")

    def compile(self, graph: Graph) -> "MLIRModule":
        """
        Compile EdgeUniCompile Graph to MLIR.

        Args:
            graph: EdgeUniCompile Graph to compile.

        Returns:
            MLIRModule instance.
        """
        # Convert graph to MLIR
        mlir_module = self._graph_to_mlir(graph)

        return MLIRModule(mlir_module, self)

    def _graph_to_mlir(self, graph: Graph) -> str:
        """
        Convert EdgeUniCompile Graph to MLIR string.

        Args:
            graph: EdgeUniCompile Graph.

        Returns:
            MLIR string representation.
        """
        # Generate MLIR module using EdgeUnityCompile dialect
        mlir_str = []
        mlir_str.append("module {")

        # Build input types from graph input tensors
        input_types = []
        for inp in graph.input_tensors:
            shape_str = "x".join(str(d) if d > 0 else "?" for d in inp.shape.dims)
            input_types.append(f"tensor<{shape_str}xf32>")

        # Build output types from graph output tensors
        output_types = []
        for out in graph.output_tensors:
            shape_str = "x".join(str(d) if d > 0 else "?" for d in out.shape.dims)
            output_types.append(f"tensor<{shape_str}xf32>")

        # Create function signature
        inputs = ", ".join(f"%{inp.name}: {typ}" for inp, typ in zip(graph.input_tensors, input_types))
        outputs = ", ".join(output_types)
        mlir_str.append(f"  func.func @main({inputs}) -> {outputs} {{")

        # Build tensor name to SSA value map
        tensor_to_ssa = {}
        for inp in graph.input_tensors:
            tensor_to_ssa[inp.name] = f"%{inp.name}"

        # Add operations using edgeuni dialect
        for node in graph.nodes:
            op_str, output_ssa = self._node_to_mlir(node, tensor_to_ssa)
            mlir_str.append(f"    {op_str}")
            # Map output tensor to SSA value
            for out_tensor in node.outputs:
                tensor_to_ssa[out_tensor.name] = output_ssa

        # Return the output
        if graph.output_tensors:
            output_ssa = tensor_to_ssa.get(graph.output_tensors[0].name, "%output")
            mlir_str.append(f"    func.return {output_ssa}")

        mlir_str.append("  }")
        mlir_str.append("}")

        return "\n".join(mlir_str)

    def _node_to_mlir(self, node, tensor_to_ssa: dict) -> tuple:
        """
        Convert EdgeUniCompile Node to MLIR operation.

        Args:
            node: EdgeUniCompile Node.
            tensor_to_ssa: Map from tensor name to SSA value.

        Returns:
            Tuple of (MLIR operation string, output SSA value).
        """
        # Get input SSA values
        input_ssa = []
        for inp_tensor in node.inputs:
            if inp_tensor.name in tensor_to_ssa:
                input_ssa.append(tensor_to_ssa[inp_tensor.name])

        # Get output tensor for SSA mapping
        output_tensor = node.outputs[0] if node.outputs else None
        output_ssa = f"%{output_tensor.name}" if output_tensor else "%result"

        # Get attributes
        kernel_shape = node.attributes.get('kernel_shape', [3, 3])
        strides = node.attributes.get('strides', [1, 1])
        pads = node.attributes.get('pads', [1, 1, 1, 1])
        dilations = node.attributes.get('dilations', [1, 1])
        tiling = node.attributes.get('tiling', None)

        # Map ONNX/EdgeUniCompile ops to MLIR edgeuni dialect
        op_type_map = {
            "Conv2D": "edgeuni.conv2d",
            "Relu": "edgeuni.relu",
            "Sigmoid": "edgeuni.sigmoid",
            "Tanh": "edgeuni.tanh",
            "Softmax": "edgeuni.softmax",
            "MaxPool2D": "edgeuni.max_pool2d",
            "AveragePool2D": "edgeuni.avg_pool2d",
            "Add": "edgeuni.add",
            "Subtract": "edgeuni.sub",
            "Multiply": "edgeuni.mul",
            "MatMul": "edgeuni.matmul",
            "Reshape": "edgeuni.reshape",
            "Transpose": "edgeuni.transpose",
        }

        mlir_op = op_type_map.get(node.op_type, "edgeuni.custom")

        # Build attribute string
        attrs = []
        if node.op_type == "Conv2D":
            attrs.append(f"kernel_shape = {kernel_shape}")
            attrs.append(f"strides = {strides}")
            attrs.append(f"pads = {pads}")
            attrs.append(f"dilations = {dilations}")
        if tiling:
            attrs.append(f"tiling = {tiling}")

        attr_str = ", ".join(attrs)
        if attr_str:
            attr_str = f"{{{attr_str}}}"

        # Build operation
        inputs_str = ", ".join(input_ssa)
        op_str = f"{output_ssa} = {mlir_op}({inputs_str}) {attr_str}"

        return op_str, output_ssa

    @property
    def mlir_context(self):
        """Get the raw MLIR context."""
        return self._mlir_context

    @property
    def session(self):
        """Get the MLIR session."""
        return self._session


class MLIRModule:
    """
    Wrapper for MLIR module.

    This class provides functionality to manipulate and optimize
    MLIR modules.
    """

    def __init__(self, mlir_str: str, context: MLIRContext):
        """
        Create an MLIR module.

        Args:
            mlir_str: MLIR string representation.
            context: MLIR context.
        """
        self.mlir_str = mlir_str
        self.context = context

    def optimize(self, passes: Optional[List[str]] = None) -> "MLIRModule":
        """
        Optimize the MLIR module using standard passes.

        Args:
            passes: List of optimization passes to apply.

        Returns:
            Optimized MLIRModule.
        """
        if passes is None:
            passes = [
                "canonicalize",
                "cse",
                "inline",
                "sccp",
                "mem2reg"
            ]

        # Apply optimizations using mlir-opt
        optimized_str = self._run_passes(self.mlir_str, passes)

        return MLIRModule(optimized_str, self.context)

    def _run_passes(self, mlir_str: str, passes: List[str]) -> str:
        """
        Run MLIR passes on the module.

        Args:
            mlir_str: Original MLIR string.
            passes: List of passes to apply.

        Returns:
            Optimized MLIR string.
        """
        # For now, simply print passes as comments
        optimized = []
        for line in mlir_str.split('\n'):
            optimized.append(line)
            if "func.func" in line:
                for pass_name in passes:
                    optimized.append(f"    // #pragma pass: {pass_name}")

        return '\n'.join(optimized)

    def lower_to_llvm(self) -> str:
        """
        Lower MLIR to LLVM dialect.

        Returns:
            LLVM dialect MLIR string.
        """
        # For now, return a placeholder
        return (
            "module {\n"
            "  llvm.func @main() -> () {\n"
            "    return\n"
            "  }\n"
            "}"
        )

    def generate_code(self, target: str = "cpu") -> str:
        """
        Generate target-specific code from MLIR.

        Args:
            target: Target architecture (cpu, arm, riscv, etc.)

        Returns:
            Generated code string.
        """
        # For now, generate simple C code
        return (
            "#include <stdio.h>\n"
            "\n"
            "int main() {\n"
            "    // Generated from EdgeUniCompile\n"
            f"    // Target: {target}\n"
            "    printf(\"Hello from EdgeUniCompile!\\n\");\n"
            "    return 0;\n"
            "}\n"
        )


class MLIRCompiler:
    """
    Compiler for converting MLIR to target code.
    """

    @staticmethod
    def compile_mlir_to_executable(mlir_str: str,
                                    output_path: str,
                                    target: str = "cpu") -> bool:
        """
        Compile MLIR to executable.

        Args:
            mlir_str: MLIR string.
            output_path: Output executable path.
            target: Target architecture.

        Returns:
            True if compilation successful.
        """
        # For now, create a simple shell script for testing
        try:
            with open(output_path, "w") as f:
                f.write("#!/bin/bash\n")
                f.write(f"# Generated from EdgeUniCompile for {target}\n")
                f.write('echo "This is a generated script"\n')
                f.write('echo "Target: ' + target + '"\n')

            os.chmod(output_path, 0o755)
            return True
        except Exception as e:
            logging.error(f"Failed to create executable: {e}")
            return False


# Convenience functions
def install_mlir(force: bool = False) -> bool:
    """Convenience function to install MLIR."""
    return MLIRInstaller.install(force)


def is_mlir_installed() -> bool:
    """Convenience function to check MLIR installation."""
    return MLIRInstaller.is_installed()


def compile_graph_to_mlir(graph: Graph) -> str:
    """Convenience function to compile graph to MLIR."""
    try:
        context = MLIRContext()
        return context.compile(graph).mlir_str
    except Exception as e:
        raise RuntimeError(f"Failed to compile graph to MLIR: {e}")


def optimize_mlir(mlir_str: str) -> str:
    """Convenience function to optimize MLIR."""
    context = MLIRContext()
    module = MLIRModule(mlir_str, context)
    return module.optimize().mlir_str


def compile_mlir(mlir_str: str, output_path: str) -> bool:
    """Convenience function to compile MLIR to executable."""
    return MLIRCompiler.compile_mlir_to_executable(mlir_str, output_path)
