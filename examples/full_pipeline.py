#!/usr/bin/env python3.13
"""
Full pipeline test implementing the plan from docs/plan.md

1. Create a single-conv ONNX model with memory > 3MB
2. Convert the model to FlatBuffer format
3. Compiler accepts the FlatBuffer IR
4. Run Python/C++ passes for tiling based on 3MB SRAM limit
5. Export IR to FlatBuffer after each pass
6. Convert FlatBuffer IR to MLIR dialect
7. Export MLIR dialect
8. End
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import edgeunicompile as euc
from edgeunicompile.onnx import ONNXConverter
from edgeunicompile.flatbuf import FlatBufferBuilder
from edgeunicompile.mlir import MLIRContext, MLIRInstaller
from edgeunicompile.passes import PassBase, PassManager
from edgeunicompile.core import Context, Status


class ExportPass(PassBase):
    """Export graph to FlatBuffer after pass execution."""

    def __init__(self, output_path: str):
        super().__init__("export_pass")
        self.output_path = output_path

    def run(self, graph, context):
        # Export graph to FlatBuffer
        FlatBufferBuilder.save_to_file(graph, self.output_path)
        file_size = os.path.getsize(self.output_path)
        print(f"  Exported graph to {self.output_path} ({file_size} bytes)")
        return Status.ok()


def step1_create_onnx_model(output_path: str = "single_conv_model.onnx"):
    """Step 1: Create a single-conv ONNX model with memory > 3MB."""
    print("\n" + "=" * 60)
    print("Step 1: Create ONNX model (memory > 3MB)")
    print("=" * 60)

    import onnx
    from onnx import numpy_helper
    import numpy as np

    # Create the ONNX model
    model = onnx.ModelProto()
    model.ir_version = 7
    model.opset_import.append(onnx.OperatorSetIdProto())
    model.opset_import[0].version = 13

    # Create graph
    graph = model.graph
    graph.name = "single_conv_graph"

    # Input tensor (NCHW: 1x3x224x224 = ~600KB)
    input_tensor = onnx.ValueInfoProto()
    input_tensor.name = "input"
    input_tensor.type.tensor_type.elem_type = onnx.TensorProto.FLOAT
    input_tensor.type.tensor_type.shape.dim.extend([
        onnx.TensorShapeProto.Dimension(dim_value=1),
        onnx.TensorShapeProto.Dimension(dim_value=3),
        onnx.TensorShapeProto.Dimension(dim_value=224),
        onnx.TensorShapeProto.Dimension(dim_value=224)
    ])
    graph.input.append(input_tensor)

    # Weights tensor (32 filters, 3x3 kernel)
    weights = np.random.randn(32, 3, 3, 3).astype(np.float32)
    weights_tensor = numpy_helper.from_array(weights, "weights")
    graph.initializer.append(weights_tensor)

    # Bias tensor
    bias = np.random.randn(32).astype(np.float32)
    bias_tensor = numpy_helper.from_array(bias, "bias")
    graph.initializer.append(bias_tensor)

    # Output tensor (1x32x224x224 = ~6.4MB)
    output_tensor = onnx.ValueInfoProto()
    output_tensor.name = "output"
    output_tensor.type.tensor_type.elem_type = onnx.TensorProto.FLOAT
    output_tensor.type.tensor_type.shape.dim.extend([
        onnx.TensorShapeProto.Dimension(dim_value=1),
        onnx.TensorShapeProto.Dimension(dim_value=32),
        onnx.TensorShapeProto.Dimension(dim_value=224),
        onnx.TensorShapeProto.Dimension(dim_value=224)
    ])
    graph.output.append(output_tensor)

    # Conv node
    conv_node = onnx.NodeProto()
    conv_node.name = "conv"
    conv_node.op_type = "Conv"
    conv_node.input.extend(["input", "weights", "bias"])
    conv_node.output.extend(["output"])
    conv_node.attribute.append(onnx.helper.make_attribute("kernel_shape", [3, 3]))
    conv_node.attribute.append(onnx.helper.make_attribute("strides", [1, 1]))
    conv_node.attribute.append(onnx.helper.make_attribute("pads", [1, 1, 1, 1]))
    conv_node.attribute.append(onnx.helper.make_attribute("dilations", [1, 1]))
    conv_node.attribute.append(onnx.helper.make_attribute("group", 1))
    graph.node.append(conv_node)

    # Validate and save
    onnx.checker.check_model(model)
    onnx.save(model, output_path)

    # Calculate memory usage
    input_size = 1 * 3 * 224 * 224 * 4  # float32
    output_size = 1 * 32 * 224 * 224 * 4
    total_io_memory = (input_size + output_size) / (1024 * 1024)

    print(f"Model saved to: {output_path}")
    print(f"Input shape: [1, 3, 224, 224]")
    print(f"Output shape: [1, 32, 224, 224]")
    print(f"I/O Memory usage: ~{total_io_memory:.1f}MB (> 3MB requirement met)")

    return model


def step2_convert_to_flatbuffer(onnx_path: str, fb_path: str):
    """Step 2: Convert ONNX model to FlatBuffer format."""
    print("\n" + "=" * 60)
    print("Step 2: Convert ONNX to FlatBuffer")
    print("=" * 60)

    import onnx
    onnx_model = onnx.load(onnx_path)

    context = Context()
    graph = ONNXConverter.convert(onnx_model, context)

    print(f"Graph name: {graph.name}")
    print(f"Nodes: {len(graph.nodes)}")
    print(f"Tensors: {len(graph.tensors)}")

    FlatBufferBuilder.save_to_file(graph, fb_path)
    file_size = os.path.getsize(fb_path)
    print(f"FlatBuffer saved to: {fb_path} ({file_size} bytes)")

    return graph, context


def step3_load_flatbuffer(fb_path: str):
    """Step 3: Compiler accepts FlatBuffer IR."""
    print("\n" + "=" * 60)
    print("Step 3: Load FlatBuffer IR")
    print("=" * 60)

    context = Context()
    graph = FlatBufferBuilder.load_from_file(fb_path)

    print(f"Loaded graph: {graph.name}")
    print(f"Nodes: {len(graph.nodes)}")
    print(f"Tensors: {len(graph.tensors)}")

    for node in graph.nodes:
        print(f"  Node: {node.name} ({node.op_type})")

    return graph, context


def step4_run_tiling_pass(graph, context, output_fb_path: str = "tiling_output.fb"):
    """Step 4: Run tiling pass based on 3MB SRAM limit."""
    print("\n" + "=" * 60)
    print("Step 4: Run Tiling Pass (3MB SRAM limit)")
    print("=" * 60)

    from edgeunicompile.passes.tiling_pass import TilingPass

    # Create context with 3MB SRAM limit
    context = Context(sram_size=3 * 1024 * 1024)
    print(f"SRAM limit: {context.sram_size / (1024 * 1024):.1f}MB")

    # Create pass manager
    pm = PassManager(context)
    pm.add_pass(TilingPass(tile_size=(64, 64)))
    pm.add_pass(ExportPass(output_fb_path))

    print(f"Running {len(pm.passes)} passes...")

    optimized_graph = pm.run(graph)

    print(f"Context counters: {context.get_all_counters()}")

    # Check tiling application
    for node in optimized_graph.nodes:
        if node.has_attribute("tiling"):
            print(f"Tiling applied to {node.name}: {node.attributes['tiling']}")

    return optimized_graph, context


def step5_export_after_passes(graph, context, output_path: str):
    """Step 5: Export IR to FlatBuffer after passes."""
    print("\n" + "=" * 60)
    print("Step 5: Export IR after passes")
    print("=" * 60)

    FlatBufferBuilder.save_to_file(graph, output_path)
    file_size = os.path.getsize(output_path)
    print(f"Exported to: {output_path} ({file_size} bytes)")

    return output_path


def step6_convert_to_mlir(graph, context):
    """Step 6: Convert FlatBuffer IR to MLIR dialect."""
    print("\n" + "=" * 60)
    print("Step 6: Convert to MLIR Dialect")
    print("=" * 60)

    # Check if MLIR is installed
    if not MLIRInstaller.is_installed():
        print("Note: MLIR not installed, using mock implementation")

    mlir_context = MLIRContext(context)
    mlir_module = mlir_context.compile(graph)

    print("MLIR module created successfully!")
    print(f"MLIR output:\n{mlir_module.mlir_str}")

    return mlir_module


def step7_export_mlir(mlir_module, output_path: str = "output.mlir"):
    """Step 7: Export MLIR dialect to file."""
    print("\n" + "=" * 60)
    print("Step 7: Export MLIR")
    print("=" * 60)

    # Optimize MLIR
    optimized_module = mlir_module.optimize()

    with open(output_path, "w") as f:
        f.write(optimized_module.mlir_str)

    file_size = os.path.getsize(output_path)
    print(f"MLIR exported to: {output_path} ({file_size} bytes)")

    # Also try to lower to LLVM and generate code
    llvm_ir = optimized_module.lower_to_llvm()
    print(f"\nLLIR dialect output:\n{llvm_ir}")

    return output_path


def main():
    """Run the full pipeline."""
    print("\n" + "#" * 60)
    print("# EdgeUniCompile Full Pipeline Test")
    print("# Based on docs/plan.md")
    print("#" * 60)

    # Define output paths
    onnx_path = "single_conv_model.onnx"
    input_fb_path = "single_conv_model.fb"
    tiling_fb_path = "tiling_output.fb"
    final_fb_path = "final_output.fb"
    mlir_path = "output.mlir"

    # Step 1: Create ONNX model
    step1_create_onnx_model(onnx_path)

    # Step 2: Convert to FlatBuffer
    graph, context = step2_convert_to_flatbuffer(onnx_path, input_fb_path)

    # Step 3: Load FlatBuffer IR
    graph, context = step3_load_flatbuffer(input_fb_path)

    # Step 4: Run tiling pass
    optimized_graph, context = step4_run_tiling_pass(graph, context, tiling_fb_path)

    # Step 5: Export after passes
    step5_export_after_passes(optimized_graph, context, final_fb_path)

    # Step 6: Convert to MLIR
    mlir_module = step6_convert_to_mlir(optimized_graph, context)

    # Step 7: Export MLIR
    step7_export_mlir(mlir_module, mlir_path)

    # Step 8: End
    print("\n" + "=" * 60)
    print("Step 8: Pipeline Completed!")
    print("=" * 60)
    print("\nGenerated files:")
    for path in [onnx_path, input_fb_path, tiling_fb_path, final_fb_path, mlir_path]:
        if os.path.exists(path):
            size = os.path.getsize(path)
            print(f"  {path}: {size} bytes")

    print("\nAll steps completed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
