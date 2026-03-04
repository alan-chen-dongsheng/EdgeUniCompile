#!/usr/bin/env python3.13

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import edgeunicompile as euc
from edgeunicompile.passes import PassBase, PassManager
from edgeunicompile.core import Status


class CountNodesPass(PassBase):
    """Count the number of nodes in the graph."""

    def __init__(self):
        super().__init__("count_nodes_pass")

    def run(self, graph, context):
        print(f"Graph '{graph.name}' has {len(graph.nodes)} nodes and {len(graph.tensors)} tensors")
        return Status.ok()


class PrintNodesPass(PassBase):
    """Print all nodes in the graph."""

    def __init__(self):
        super().__init__("print_nodes_pass")

    def run(self, graph, context):
        print("Nodes:")
        for node in graph.nodes:
            print(f"  {node}")
        return Status.ok()


class PrintTensorsPass(PassBase):
    """Print all tensors in the graph."""

    def __init__(self):
        super().__init__("print_tensors_pass")

    def run(self, graph, context):
        print("Tensors:")
        for tensor in graph.tensors:
            print(f"  {tensor}")
        return Status.ok()


def main():
    # Load graph from FlatBuffer
    input_path = sys.argv[1]

    print("EdgeUniCompile - Pass Testing")
    print("=" * 50)

    print(f"Loading graph from {input_path}...")
    graph = euc.FlatBufferBuilder.load_from_file(input_path)

    # Create context and pass manager
    context = euc.Context()
    pm = euc.PassManager(context)
    pm.add_pass(CountNodesPass())
    pm.add_pass(PrintNodesPass())
    pm.add_pass(PrintTensorsPass())

    print(f"\nPass manager created with {len(pm.passes)} passes")

    # Run passes
    print("\nRunning passes...")
    try:
        optimized_graph = pm.run(graph)
        print("\nPasses completed successfully")
        print(f"Nodes after passes: {len(optimized_graph.nodes)}")
        print(f"Tensors after passes: {len(optimized_graph.tensors)}")
    except Exception as e:
        print(f"\nPasses failed: {e}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python test_passes.py <graph.fb>")
        sys.exit(1)
    main()
