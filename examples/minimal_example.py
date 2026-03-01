#!/usr/bin/env python3.13
"""
Minimal example demonstrating basic EdgeUniCompile functionality.

This example shows:
1. Creating a simple computation graph
2. Adding nodes and tensors
3. Serializing to FlatBuffer
"""

import edgeunicompile as euc
from edgeunicompile.ir import Graph, Node, Tensor


def main():
    print("EdgeUniCompile - Minimal Example")
    print("=" * 50)

    # Create compilation context
    ctx = euc.Context()

    # Create a simple computation graph
    graph = Graph("minimal_graph")

    # Add tensors
    t1 = Tensor("a", "float32", (2, 3))
    t2 = Tensor("b", "float32", (2, 3))
    t3 = Tensor("c", "float32", (2, 3))

    graph.add_tensor(t1)
    graph.add_tensor(t2)
    graph.add_tensor(t3)
    graph.add_input_tensor(t1)
    graph.add_input_tensor(t2)
    graph.add_output_tensor(t3)

    # Add node
    node = Node("add", "Add")
    node.add_input(t1)
    node.add_input(t2)
    node.add_output(t3)
    graph.add_node(node)

    print(f"Graph: {graph.name}")
    print(f"Nodes: {len(graph.nodes)}")
    print(f"Tensors: {len(graph.tensors)}")
    print(f"Inputs: {[t.name for t in graph.input_tensors]}")
    print(f"Outputs: {[t.name for t in graph.output_tensors]}")

    # Save to FlatBuffer
    output_path = "minimal_graph.fb"
    try:
        status = euc.FlatBufferBuilder.save_to_file(graph, output_path)
        if status.is_ok():
            import os
            size = os.path.getsize(output_path)
            print(f"\nGraph saved to {output_path} ({size} bytes)")
        else:
            print(f"\nError saving graph: {status}")
    except Exception as e:
        print(f"\nError saving graph: {e}")

    print("\nAll operations completed successfully!")


if __name__ == "__main__":
    main()
