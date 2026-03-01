#!/usr/bin/env python3.13
"""
Simple example demonstrating EdgeUniCompile usage.

This example shows:
1. Creating a simple computation graph
2. Applying passes
3. Compiling to MLIR
4. Generating target code
"""

import edgeunicompile as euc
from edgeunicompile.ir import Graph, Node, Tensor


def main():
    print("EdgeUniCompile - Simple Example")
    print("=" * 50)

    # 1. Create compilation context with SRAM size constraint
    ctx = euc.Context(
        opt_level=3,
        sram_size=16 * 1024 * 1024,  # 16MB SRAM
        target_arch="armv8",
        verbose_mode=True
    )

    print(f"\nContext created:")
    print(f"  Optimization level: {ctx.opt_level}")
    print(f"  SRAM size: {ctx.sram_size / (1024*1024)}MB")
    print(f"  Target architecture: {ctx.target_arch}")
    print(f"  Verbose mode: {ctx.verbose_mode}")

    # 2. Create a simple computation graph
    graph = Graph("simple_add_graph")

    # Add tensors
    t1 = Tensor("input1", "float32", (2, 3))
    t2 = Tensor("input2", "float32", (2, 3))
    t3 = Tensor("output", "float32", (2, 3))

    graph.add_tensor(t1)
    graph.add_tensor(t2)
    graph.add_tensor(t3)
    graph.add_input_tensor(t1)
    graph.add_input_tensor(t2)
    graph.add_output_tensor(t3)

    # Add node
    node = Node("add_node", "Add")
    node.add_input(t1)
    node.add_input(t2)
    node.add_output(t3)
    graph.add_node(node)

    print(f"\nGraph created:")
    print(f"  Name: {graph.name}")
    print(f"  Nodes: {len(graph.nodes)}")
    print(f"  Tensors: {len(graph.tensors)}")
    print(f"  Inputs: {[t.name for t in graph.input_tensors]}")
    print(f"  Outputs: {[t.name for t in graph.output_tensors]}")

    # 3. Create and configure pass manager
    pass_manager = euc.PassManager(ctx)

    print(f"\nPass manager created with {len(pass_manager.passes)} passes:")
    for pass_info in pass_manager.list_passes():
        status = "Enabled" if pass_info["enabled"] else "Disabled"
        print(f"  - {pass_info['name']} ({pass_info['type']}) - {status}")

    # 4. Run passes on the graph
    print("\nRunning passes...")
    try:
        optimized_graph = pass_manager.run(graph)
        print(f"Passes completed successfully")
        print(f"  Nodes after optimization: {len(optimized_graph.nodes)}")
        print(f"  Tensors after optimization: {len(optimized_graph.tensors)}")
    except Exception as e:
        print(f"Error running passes: {e}")
        return

    # 5. Compile to MLIR
    print("\nCompiling to MLIR...")
    try:
        mlir_context = euc.MLIRContext(ctx)
        mlir_module = mlir_context.compile(optimized_graph)
        print("MLIR compilation completed successfully")
    except Exception as e:
        print(f"Error compiling to MLIR: {e}")
        return

    # 6. Optimize MLIR
    print("\nOptimizing MLIR...")
    try:
        optimized_mlir = mlir_module.optimize()
        print("MLIR optimization completed successfully")
    except Exception as e:
        print(f"Error optimizing MLIR: {e}")
        return

    # 7. Generate target code
    print("\nGenerating target code...")
    try:
        code = optimized_mlir.generate_code("armv8")
        print("Code generation completed successfully")
        print("\nGenerated ARMv8 code:")
        print("=" * 50)
        print(code)
    except Exception as e:
        print(f"Error generating code: {e}")
        return

    # 8. Save to FlatBuffer
    print("\nSaving to FlatBuffer...")
    try:
        output_path = "simple_add_graph.fb"
        status = euc.FlatBufferBuilder.save_to_file(optimized_graph, output_path)
        if status.is_ok():
            import os
            size = os.path.getsize(output_path)
            print(f"Graph saved to {output_path} ({size} bytes)")
        else:
            print(f"Error saving graph: {status}")
    except Exception as e:
        print(f"Error saving graph: {e}")

    print("\n" + "=" * 50)
    print("All operations completed successfully!")


if __name__ == "__main__":
    main()
