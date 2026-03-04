#!/usr/bin/env python3.13
"""
Convert EdgeUniCompile FlatBuffer to ONNX format for Netron visualization.

Usage:
    python fb2onnx.py <input.fb> [output.onnx]
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import edgeunicompile as euc
from edgeunicompile.ir import Graph, Node, Tensor
from edgeunicompile.core import DataType, Shape
import onnx
from onnx import numpy_helper, helper
import numpy as np


class FlatBufferToONNXConverter:
    """
    Converter for EdgeUniCompile FlatBuffer to ONNX model.

    This class provides functionality to convert EdgeUniCompile Graph IR
    back to ONNX format for visualization in Netron.
    """

    # Mapping from EdgeUniCompile op types to ONNX op types
    EUC_TO_ONNX_OP = {
        "Conv2D": "Conv",
        "MaxPool2D": "MaxPool",
        "AveragePool2D": "AveragePool",
        "Relu": "Relu",
        "Sigmoid": "Sigmoid",
        "Tanh": "Tanh",
        "Softmax": "Softmax",
        "Add": "Add",
        "Subtract": "Sub",
        "Multiply": "Mul",
        "Divide": "Div",
        "MatMul": "MatMul",
        "Reshape": "Reshape",
        "Transpose": "Transpose",
        "Flatten": "Flatten",
        "Gemm": "Gemm",
        "BatchNorm": "BatchNormalization",
        "Concat": "Concat",
        "Split": "Split",
        "Pad": "Pad",
        "GlobalAveragePool": "GlobalAveragePool",
        "GlobalMaxPool": "GlobalMaxPool",
    }

    # Mapping from DataType to ONNX tensor element type
    DATA_TYPE_TO_ONNX = {
        DataType.FLOAT32: onnx.TensorProto.FLOAT,
        DataType.FLOAT16: onnx.TensorProto.FLOAT16,
        DataType.INT32: onnx.TensorProto.INT32,
        DataType.INT16: onnx.TensorProto.INT16,
        DataType.INT8: onnx.TensorProto.INT8,
        DataType.UINT32: onnx.TensorProto.UINT32,
        DataType.UINT16: onnx.TensorProto.UINT16,
        DataType.UINT8: onnx.TensorProto.UINT8,
        DataType.BOOL: onnx.TensorProto.BOOL,
        DataType.COMPLEX64: onnx.TensorProto.COMPLEX64,
    }

    @staticmethod
    def convert(graph: Graph, model_name: str = "edgeunicompile_model") -> onnx.ModelProto:
        """
        Convert EdgeUniCompile Graph to ONNX model.

        Args:
            graph: EdgeUniCompile Graph to convert.
            model_name: Name for the ONNX model.

        Returns:
            ONNX ModelProto.
        """
        # Create ONNX model
        model = onnx.ModelProto()
        model.ir_version = 7
        model.producer_name = "EdgeUniCompile"
        model.producer_version = euc.__version__
        model.domain = "edgeunicompile"
        model.opset_import.append(onnx.OperatorSetIdProto())
        model.opset_import[0].version = 13

        # Create graph
        onnx_graph = model.graph
        onnx_graph.name = graph.name

        # Convert tensors and build name map
        tensor_name_map = {}  # EdgeUniCompile tensor name -> ONNX tensor name

        # First, handle all tensors
        for tensor in graph.tensors:
            onnx_tensor = FlatBufferToONNXConverter._convert_tensor(tensor)
            if tensor.data:
                # This is a weight/initializer
                onnx_graph.initializer.append(onnx_tensor)
            tensor_name_map[tensor.name] = tensor.name

        # Add input tensors as ValueInfoProto
        for tensor in graph.input_tensors:
            if not tensor.data:  # Only add non-constant inputs
                value_info = FlatBufferToONNXConverter._tensor_to_value_info(tensor)
                onnx_graph.input.append(value_info)

        # Add output tensors as ValueInfoProto
        for tensor in graph.output_tensors:
            value_info = FlatBufferToONNXConverter._tensor_to_value_info(tensor)
            onnx_graph.output.append(value_info)

        # Convert nodes
        for euc_node in graph.nodes:
            onnx_node = FlatBufferToONNXConverter._convert_node(euc_node, tensor_name_map)
            onnx_graph.node.append(onnx_node)

        # Validate the model
        try:
            onnx.checker.check_model(model)
        except Exception as e:
            print(f"Warning: ONNX model validation warning: {e}")

        return model

    @staticmethod
    def _convert_tensor(tensor: Tensor) -> onnx.TensorProto:
        """Convert EdgeUniCompile Tensor to ONNX TensorProto."""
        # Map data type
        onnx_dtype = FlatBufferToONNXConverter.DATA_TYPE_TO_ONNX.get(
            tensor.dtype, onnx.TensorProto.FLOAT
        )

        if tensor.data:
            # Has raw data - create initializer
            # Try to reconstruct numpy array from raw bytes
            try:
                num_elements = tensor.shape.num_elements()
                if tensor.dtype == DataType.FLOAT32:
                    data = np.frombuffer(tensor.data, dtype=np.float32, count=num_elements)
                elif tensor.dtype == DataType.FLOAT16:
                    data = np.frombuffer(tensor.data, dtype=np.float16, count=num_elements)
                elif tensor.dtype == DataType.INT32:
                    data = np.frombuffer(tensor.data, dtype=np.int32, count=num_elements)
                elif tensor.dtype == DataType.INT8:
                    data = np.frombuffer(tensor.data, dtype=np.int8, count=num_elements)
                else:
                    # Default to float32
                    data = np.frombuffer(tensor.data, dtype=np.float32, count=num_elements)

                data = data.reshape(tensor.shape.dims)
                return numpy_helper.from_array(data, tensor.name)
            except Exception as e:
                print(f"Warning: Could not reconstruct tensor data for {tensor.name}: {e}")
                # Create empty tensor
                tensor_proto = onnx.TensorProto()
                tensor_proto.name = tensor.name
                tensor_proto.data_type = onnx_dtype
                tensor_proto.dims.extend(tensor.shape.dims)
                return tensor_proto
        else:
            # No data - create placeholder
            tensor_proto = onnx.TensorProto()
            tensor_proto.name = tensor.name
            tensor_proto.data_type = onnx_dtype
            tensor_proto.dims.extend(tensor.shape.dims)
            return tensor_proto

    @staticmethod
    def _tensor_to_value_info(tensor: Tensor) -> onnx.ValueInfoProto:
        """Convert EdgeUniCompile Tensor to ONNX ValueInfoProto."""
        value_info = onnx.ValueInfoProto()
        value_info.name = tensor.name

        # Map data type
        onnx_dtype = FlatBufferToONNXConverter.DATA_TYPE_TO_ONNX.get(
            tensor.dtype, onnx.TensorProto.FLOAT
        )

        value_info.type.tensor_type.elem_type = onnx_dtype

        # Add shape
        for dim in tensor.shape.dims:
            dim_proto = onnx.TensorShapeProto.Dimension()
            dim_proto.dim_value = dim
            value_info.type.tensor_type.shape.dim.append(dim_proto)

        return value_info

    @staticmethod
    def _convert_node(euc_node: Node, tensor_name_map: dict) -> onnx.NodeProto:
        """Convert EdgeUniCompile Node to ONNX NodeProto."""
        # Map op type
        onnx_op_type = FlatBufferToONNXConverter.EUC_TO_ONNX_OP.get(
            euc_node.op_type, euc_node.op_type
        )

        # Create node
        onnx_node = onnx.NodeProto()
        onnx_node.name = euc_node.name
        onnx_node.op_type = onnx_op_type

        # Add inputs
        for inp_tensor in euc_node.inputs:
            onnx_node.input.append(tensor_name_map.get(inp_tensor.name, inp_tensor.name))

        # Add outputs
        for out_tensor in euc_node.outputs:
            onnx_node.output.append(tensor_name_map.get(out_tensor.name, out_tensor.name))

        # Convert attributes
        attrs = FlatBufferToONNXConverter._convert_attributes(euc_node, onnx_op_type)
        onnx_node.attribute.extend(attrs)

        return onnx_node

    @staticmethod
    def _convert_attributes(euc_node: Node, onnx_op_type: str) -> list:
        """Convert EdgeUniCompile node attributes to ONNX attributes."""
        attrs = []

        # Standard ONNX attributes to keep (operator-specific)
        standard_attrs = {
            # Conv
            "kernel_shape", "strides", "pads", "dilations", "group",
            # Pool
            "kernel_shape", "strides", "pads", "ceil_mode",
            # Concat
            "axis",
            # Reshape
            "shape", "allowzero",
            # Transpose
            "perm",
            # Softmax
            "axis",
            # Gemm
            "alpha", "beta", "transA", "transB",
            # BatchNorm
            "epsilon", "momentum", "spatial",
        }

        # Tiling-specific attributes to skip (not standard ONNX)
        tiling_attrs = {
            "tiling", "tile_index", "tile_position", "tile_output_slice",
            "tile_input_slice", "is_tiled", "concat_axis", "tile_count",
            "tiles_x", "tiles_y", "is_tiled_concat"
        }

        for attr_name, attr_value in euc_node.attributes.items():
            # Skip tiling-specific attributes (not needed for visualization)
            if attr_name in tiling_attrs:
                continue

            # Only include standard ONNX attributes
            if attr_name not in standard_attrs:
                continue

            # Convert value to ONNX attribute
            if isinstance(attr_value, int):
                attr = helper.make_attribute(attr_name, attr_value)
            elif isinstance(attr_value, float):
                attr = helper.make_attribute(attr_name, attr_value)
            elif isinstance(attr_value, str):
                attr = helper.make_attribute(attr_name, attr_value)
            elif isinstance(attr_value, list):
                if all(isinstance(x, int) for x in attr_value):
                    attr = helper.make_attribute(attr_name, attr_value)
                elif all(isinstance(x, float) for x in attr_value):
                    attr = helper.make_attribute(attr_name, attr_value)
                else:
                    continue
            else:
                continue

            attrs.append(attr)

        return attrs


def convert_file(input_path: str, output_path: str = None):
    """
    Convert a FlatBuffer file to ONNX format.

    Args:
        input_path: Path to input .fb file.
        output_path: Path to output .onnx file (default: same name with .onnx extension).

    Returns:
        Path to output file, or None on failure.
    """
    if output_path is None:
        base_name = os.path.splitext(input_path)[0]
        output_path = base_name + "_for_netron.onnx"

    print(f"EdgeUniCompile - FlatBuffer to ONNX Converter")
    print("=" * 60)
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print()

    # Load FlatBuffer
    print("Loading FlatBuffer...")
    try:
        graph = euc.FlatBufferBuilder.load_from_file(input_path)
        print(f"  Graph name: {graph.name}")
        print(f"  Nodes: {len(graph.nodes)}")
        print(f"  Tensors: {len(graph.tensors)}")
    except Exception as e:
        print(f"Error loading FlatBuffer: {e}")
        raise

    # Convert to ONNX
    print("\nConverting to ONNX...")
    try:
        onnx_model = FlatBufferToONNXConverter.convert(graph, graph.name)
        print("  Conversion successful!")
    except Exception as e:
        print(f"Error converting to ONNX: {e}")
        raise

    # Save ONNX model
    print(f"\nSaving ONNX model to {output_path}...")
    try:
        onnx.save(onnx_model, output_path)
        file_size = os.path.getsize(output_path)
        print(f"  Saved! Size: {file_size} bytes")
    except Exception as e:
        print(f"Error saving ONNX model: {e}")
        raise

    print("\n" + "=" * 60)
    print("Conversion completed!")
    print(f"\nTo visualize in Netron:")
    print(f"  1. Open Netron (https://netron.app/)")
    print(f"  2. Open file: {output_path}")
    print(f"\nOr use command line:")
    print(f"  netron {output_path}")

    return output_path


def main():
    if len(sys.argv) < 2:
        print("Usage: python fb2onnx.py <input.fb> [output.onnx]")
        print("       python fb2onnx.py --batch <input_dir> [output_dir]")
        print()
        print("Convert EdgeUniCompile FlatBuffer to ONNX format for Netron visualization.")
        print()
        print("Arguments:")
        print("  input.fb     - Input FlatBuffer file")
        print("  output.onnx  - Output ONNX file (optional, default: <input>_for_netron.onnx)")
        print()
        print("Batch mode:")
        print("  --batch <input_dir> [output_dir]")
        print("    Convert all .fb files in input_dir to output_dir")
        print("    (default output_dir: current directory)")
        sys.exit(1)

    # Check for batch mode
    if sys.argv[1] == "--batch":
        if len(sys.argv) < 3:
            print("Error: --batch requires input directory argument")
            sys.exit(1)

        input_dir = sys.argv[2]
        output_dir = sys.argv[3] if len(sys.argv) > 3 else "."

        batch_convert(input_dir, output_dir)
    else:
        input_path = sys.argv[1]
        output_path = sys.argv[2] if len(sys.argv) > 2 else None

        if not os.path.exists(input_path):
            print(f"Error: Input file not found: {input_path}")
            sys.exit(1)

        try:
            convert_file(input_path, output_path)
        except Exception as e:
            print(f"\nConversion failed: {e}")
            sys.exit(1)


def batch_convert(input_dir: str, output_dir: str):
    """
    Batch convert all FlatBuffer files in a directory.

    Args:
        input_dir: Directory containing .fb files.
        output_dir: Directory for output .onnx files.
    """
    if not os.path.isdir(input_dir):
        print(f"Error: Input directory not found: {input_dir}")
        sys.exit(1)

    # Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)

    print(f"EdgeUniCompile - Batch FlatBuffer to ONNX Converter")
    print("=" * 60)
    print(f"Input directory:  {input_dir}")
    print(f"Output directory: {output_dir}")
    print()

    # Find all .fb files
    fb_files = [f for f in os.listdir(input_dir) if f.endswith('.fb')]

    if not fb_files:
        print(f"No .fb files found in {input_dir}")
        sys.exit(0)

    print(f"Found {len(fb_files)} .fb file(s)")
    print()

    # Convert each file
    success_count = 0
    error_count = 0

    for fb_file in fb_files:
        input_path = os.path.join(input_dir, fb_file)
        base_name = os.path.splitext(fb_file)[0]
        output_path = os.path.join(output_dir, base_name + ".onnx")

        try:
            result = convert_file(input_path, output_path)
            if result:
                success_count += 1
            print()
        except Exception as e:
            print(f"Skipping {fb_file}: {e}\n")
            error_count += 1

    print()
    print("=" * 60)
    print(f"Batch conversion completed!")
    print(f"  Successful: {success_count}")
    print(f"  Failed: {error_count}")


if __name__ == "__main__":
    main()
