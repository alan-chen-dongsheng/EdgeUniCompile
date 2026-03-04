#!/usr/bin/env python3.13
"""
Simple example demonstrating EdgeUniCompile usage with Conv2D tiling.

This example shows:
1. Creating a Conv2D model with memory > 3MB
2. Applying tiling pass to split Conv2D into multiple tiles
3. Exporting IR after each pass (FlatBuffer format)
4. Compiling to MLIR
5. Generating target code

The Conv2D operation produces output of [1, 32, 224, 224] = ~6.4MB
With a SRAM limit of 3MB, this requires tiling.
"""

import edgeunicompile as euc
from edgeunicompile.ir import Graph, Node, Tensor
from edgeunicompile.core import Context, Status, DataType, Shape
from edgeunicompile.passes import PassBase, PassManager
import os
import numpy as np

# Output directory for all IR files
OUTPUT_DIR = "simple_example_output"

# SRAM limit: 3MB (as per plan.md requirements)
SRAM_LIMIT = 3 * 1024 * 1024


class ExportPass(PassBase):
    """Export graph to FlatBuffer after pass execution."""

    def __init__(self, output_path: str):
        super().__init__("export_pass")
        self.output_path = output_path

    def run(self, graph, context):
        # Export graph to FlatBuffer
        euc.FlatBufferBuilder.save_to_file(graph, self.output_path)
        file_size = os.path.getsize(self.output_path)
        print(f"  Exported to {self.output_path} ({file_size} bytes)")
        return Status.ok()


def export_graph(graph, filename, description):
    """Helper function to export graph to FlatBuffer."""
    filepath = os.path.join(OUTPUT_DIR, filename)
    status = euc.FlatBufferBuilder.save_to_file(graph, filepath)
    if not status.is_ok():
        print(f"  Error exporting {description}: {status}")
        return None
    file_size = os.path.getsize(filepath)
    print(f"  {description}: {filepath} ({file_size} bytes)")
    return filepath


def create_conv_graph() -> Graph:
    """
    Create a Conv2D graph with memory usage > 3MB.

    Model architecture:
    - Input: [1, 3, 224, 224] = ~600KB
    - Conv2D with 32 filters, 3x3 kernel
    - Output: [1, 32, 224, 224] = ~6.4MB

    Total I/O memory: ~7MB (exceeds 3MB SRAM limit)
    """
    graph = Graph("conv2d_graph")

    # Input tensor: NCHW format [1, 3, 224, 224]
    input_tensor = Tensor(
        name="input",
        dtype=DataType.FLOAT32,
        shape=Shape([1, 3, 224, 224])
    )
    graph.add_tensor(input_tensor)
    graph.add_input_tensor(input_tensor)

    # Weight tensor: [out_channels, in_channels, kernel_h, kernel_w] = [32, 3, 3, 3]
    # Size: 32 * 3 * 3 * 3 * 4 bytes = 3,456 bytes
    weight_data = np.random.randn(32, 3, 3, 3).astype(np.float32).tobytes()
    weight_tensor = Tensor(
        name="conv_weight",
        dtype=DataType.FLOAT32,
        shape=Shape([32, 3, 3, 3]),
        data=weight_data
    )
    graph.add_tensor(weight_tensor)

    # Bias tensor: [out_channels] = [32]
    bias_data = np.random.randn(32).astype(np.float32).tobytes()
    bias_tensor = Tensor(
        name="conv_bias",
        dtype=DataType.FLOAT32,
        shape=Shape([32]),
        data=bias_data
    )
    graph.add_tensor(bias_tensor)

    # Output tensor: [1, 32, 224, 224] = ~6.4MB
    output_tensor = Tensor(
        name="output",
        dtype=DataType.FLOAT32,
        shape=Shape([1, 32, 224, 224])
    )
    graph.add_tensor(output_tensor)
    graph.add_output_tensor(output_tensor)

    # Conv2D node
    conv_node = Node("conv2d", "Conv2D")
    conv_node.add_input(input_tensor)
    conv_node.add_input(weight_tensor)
    conv_node.add_input(bias_tensor)
    conv_node.add_output(output_tensor)

    # Conv2D attributes
    conv_node.set_attribute("kernel_shape", [3, 3])
    conv_node.set_attribute("strides", [1, 1])
    conv_node.set_attribute("pads", [1, 1, 1, 1])  # top, bottom, left, right
    conv_node.set_attribute("dilations", [1, 1])
    conv_node.set_attribute("group", 1)

    graph.add_node(conv_node)

    return graph


def print_memory_info(graph: Graph):
    """Print memory usage information for the graph."""
    print("\n--- Memory Usage Analysis ---")

    total_memory = 0
    for tensor in graph.tensors:
        mem_bytes = tensor.shape.num_elements() * 4  # float32 = 4 bytes
        mem_kb = mem_bytes / 1024
        mem_mb = mem_bytes / (1024 * 1024)

        if mem_mb >= 1:
            print(f"  {tensor.name}: {tensor.shape.dims} = {mem_mb:.2f}MB")
        else:
            print(f"  {tensor.name}: {tensor.shape.dims} = {mem_kb:.1f}KB")

        total_memory += mem_bytes

    print(f"\n  Total tensor memory: {total_memory / (1024*1024):.2f}MB")
    print(f"  SRAM limit: {SRAM_LIMIT / (1024*1024):.1f}MB")

    if total_memory > SRAM_LIMIT:
        print(f"  Status: EXCEEDS SRAM - Tiling required!")
    else:
        print(f"  Status: Fits in SRAM")


def main():
    print("EdgeUniCompile - Conv2D Tiling Example")
    print("=" * 60)

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"\nOutput directory: {OUTPUT_DIR}")

    # 1. Create compilation context with 3MB SRAM limit
    ctx = Context(
        opt_level=3,
        sram_size=SRAM_LIMIT,  # 3MB SRAM - requires tiling
        target_arch="armv8",
        verbose_mode=True
    )

    print(f"\nContext created:")
    print(f"  Optimization level: {ctx.opt_level}")
    print(f"  SRAM size: {ctx.sram_size / (1024*1024):.1f}MB")
    print(f"  Target architecture: {ctx.target_arch}")
    print(f"  Verbose mode: {ctx.verbose_mode}")

    # 2. Create Conv2D computation graph
    graph = create_conv_graph()

    print(f"\nGraph created:")
    print(f"  Name: {graph.name}")
    print(f"  Nodes: {len(graph.nodes)}")
    print(f"  Tensors: {len(graph.tensors)}")
    print(f"  Inputs: {[t.name for t in graph.input_tensors]}")
    print(f"  Outputs: {[t.name for t in graph.output_tensors]}")

    # Print memory info
    print_memory_info(graph)

    # Export initial graph
    print("\n--- Exporting Initial IR ---")
    export_graph(graph, "00_initial_graph.fb", "Initial graph")

    # 3. Create pass manager
    pass_manager = PassManager(ctx)

    print(f"\nPass manager created:")

    # 4. Add and run tiling pass
    print("\n--- Running Tiling Pass ---")
    from edgeunicompile.passes.tiling_pass import TilingPass

    # Use 64x64 tile size
    tiling_pass = TilingPass(tile_size=(64, 64))

    # Export after tiling
    export_after_tiling = ExportPass(os.path.join(OUTPUT_DIR, "01_after_tiling.fb"))

    pass_manager.add_pass(tiling_pass)
    pass_manager.add_pass(export_after_tiling)

    try:
        optimized_graph = pass_manager.run(graph)
        print(f"\nTiling pass completed successfully")
        print(f"  Nodes before: 1 (Conv2D)")
        print(f"  Nodes after: {len(optimized_graph.nodes)}")
        print(f"  Tensors: {len(optimized_graph.tensors)}")
    except Exception as e:
        print(f"Error running tiling pass: {e}")
        import traceback
        traceback.print_exc()
        return

    # Print tiling results
    print("\n--- Tiling Results ---")
    conv_count = sum(1 for n in optimized_graph.nodes if n.op_type == "Conv2D")
    concat_count = sum(1 for n in optimized_graph.nodes if n.op_type == "Concat")
    tiled_count = sum(1 for n in optimized_graph.nodes if n.has_attribute("is_tiled"))

    print(f"  Conv2D nodes: {conv_count}")
    print(f"  Concat nodes: {concat_count}")
    print(f"  Tiled Conv2D nodes: {tiled_count}")

    # Show tile information
    print("\n  Tile details:")
    for node in optimized_graph.nodes:
        if node.has_attribute("is_tiled"):
            tile_idx = node.get_attribute("tile_index", "?")
            tile_pos = node.get_attribute("tile_position", [0, 0])
            output_slice = node.get_attribute("tile_output_slice", [])
            print(f"    {node.name}: tile[{tile_idx}] pos={tile_pos} slice={output_slice}")

    # Export optimized graph
    print("\n--- Exporting Optimized IR ---")
    export_graph(optimized_graph, "02_after_all_passes.fb", "After all passes")

    # 5. Compile to MLIR
    print("\n--- Compiling to MLIR ---")
    try:
        mlir_context = euc.MLIRContext(ctx)
        mlir_module = mlir_context.compile(optimized_graph)
        print("MLIR compilation completed successfully")

        # Save MLIR
        mlir_path = os.path.join(OUTPUT_DIR, "03_model.mlir")
        with open(mlir_path, "w") as f:
            f.write(mlir_module.mlir_str)
        print(f"  MLIR saved to: {mlir_path} ({os.path.getsize(mlir_path)} bytes)")
    except Exception as e:
        print(f"Error compiling to MLIR: {e}")
        return

    # 6. Optimize MLIR
    print("\n--- Optimizing MLIR ---")
    try:
        optimized_mlir = mlir_module.optimize()
        print("MLIR optimization completed successfully")

        # Save optimized MLIR
        opt_mlir_path = os.path.join(OUTPUT_DIR, "04_model_optimized.mlir")
        with open(opt_mlir_path, "w") as f:
            f.write(optimized_mlir.mlir_str)
        print(f"  Optimized MLIR saved to: {opt_mlir_path} ({os.path.getsize(opt_mlir_path)} bytes)")
    except Exception as e:
        print(f"Error optimizing MLIR: {e}")
        return

    # 7. Generate target code
    print("\n--- Generating Target Code ---")
    try:
        code = optimized_mlir.generate_code("armv8")
        print("Code generation completed successfully")
        print("\nGenerated ARMv8 code:")
        print("=" * 50)
        print(code)

        # Save generated code
        code_path = os.path.join(OUTPUT_DIR, "05_generated_code.cpp")
        with open(code_path, "w") as f:
            f.write(code)
        print(f"\nGenerated code saved to: {code_path}")
    except Exception as e:
        print(f"Error generating code: {e}")
        return

    # 8. Save final FlatBuffer
    print("\n--- Saving Final FlatBuffer ---")
    try:
        output_path = os.path.join(OUTPUT_DIR, "06_final_graph.fb")
        status = euc.FlatBufferBuilder.save_to_file(optimized_graph, output_path)
        if status.is_ok():
            size = os.path.getsize(output_path)
            print(f"Graph saved to {output_path} ({size} bytes)")
        else:
            print(f"Error saving graph: {status}")
    except Exception as e:
        print(f"Error saving graph: {e}")

    # Summary
    print("\n" + "=" * 60)
    print("All operations completed successfully!")
    print(f"\nOutput files in '{OUTPUT_DIR}':")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        filepath = os.path.join(OUTPUT_DIR, f)
        size = os.path.getsize(filepath)
        file_type = ""
        if f.endswith(".fb"):
            file_type = " (FlatBuffer IR)"
        elif f.endswith(".onnx"):
            file_type = " (ONNX for Netron)"
        elif f.endswith(".mlir"):
            file_type = " (MLIR)"
        elif f.endswith(".cpp"):
            file_type = " (Generated code)"
        print(f"  {f}: {size} bytes{file_type}")

    print("\n--- Context Counters ---")
    for key, value in ctx.get_all_counters().items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
