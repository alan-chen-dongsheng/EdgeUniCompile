#!/usr/bin/env python3.13

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import edgeunicompile as euc
import pprint


def main(input_path):
    print("Checking FlatBuffer contents")
    print("=" * 50)

    try:
        with open(input_path, "rb") as f:
            buffer = f.read()

        print(f"File size: {len(buffer)} bytes")
        print()

        # Load graph from FlatBuffer
        graph = euc.FlatBufferBuilder.load_from_file(input_path)
        print(f"Graph name: {graph.name}")
        print(f"Nodes: {len(graph.nodes)}")
        print()

        print(f"Nodes:")
        for i, node in enumerate(graph.nodes):
            print(f"  {i}: {node.name} ({node.op_type})")
            print(f"    Inputs: {[t.name for t in node.inputs]}")
            print(f"    Outputs: {[t.name for t in node.outputs]}")

        print()
        print(f"Tensors:")
        for i, tensor in enumerate(graph.tensors):
            print(f"  {i}: {tensor.name} ({tensor.dtype}) {tensor.shape.dims}")
            if tensor.data:
                print(f"    Data size: {len(tensor.data)} bytes")

        print()
        print(f"Input tensors: {[t.name for t in graph.input_tensors]}")
        print(f"Output tensors: {[t.name for t in graph.output_tensors]}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        print(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python check_graph.py <flatbuffer.fb>")
        sys.exit(1)

    main(sys.argv[1])
