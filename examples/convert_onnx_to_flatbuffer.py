#!/usr/bin/env python3.13
"""
Convert ONNX model to EdgeUniCompile FlatBuffer format.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import edgeunicompile as euc
from edgeunicompile.onnx import ONNXConverter


def main(input_path, output_path):
    print("EdgeUniCompile - ONNX to FlatBuffer Converter")
    print("=" * 60)

    # Load ONNX model
    print(f"Loading ONNX model from {input_path}...")
    try:
        import onnx
        onnx_model = onnx.load(input_path)
    except Exception as e:
        print(f"Error loading ONNX model: {e}")
        sys.exit(1)

    # Convert to EdgeUniCompile Graph
    print("Converting ONNX model to EdgeUniCompile Graph...")
    try:
        context = euc.Context()
        graph = ONNXConverter.convert(onnx_model, context)
        print(f"Conversion successful! Graph name: {graph.name}")
        print(f"Nodes: {len(graph.nodes)}")
        print(f"Tensors: {len(graph.tensors)}")
        print(f"Inputs: {[t.name for t in graph.input_tensors]}")
        print(f"Outputs: {[t.name for t in graph.output_tensors]}")
    except Exception as e:
        print(f"Error converting model: {e}")
        import traceback
        print(traceback.format_exc())
        sys.exit(1)

    # Save to FlatBuffer
    print(f"Saving FlatBuffer to {output_path}...")
    try:
        status = euc.FlatBufferBuilder.save_to_file(graph, output_path)
        if status.is_ok():
            size = os.path.getsize(output_path)
            print(f"FlatBuffer saved successfully! Size: {size} bytes")
        else:
            print(f"Error saving FlatBuffer: {status}")
            sys.exit(1)
    except Exception as e:
        print(f"Error saving FlatBuffer: {e}")
        import traceback
        print(traceback.format_exc())
        sys.exit(1)

    print("Conversion completed!")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python convert_onnx_to_flatbuffer.py <input.onnx> <output.fb>")
        sys.exit(1)

    main(sys.argv[1], sys.argv[2])
