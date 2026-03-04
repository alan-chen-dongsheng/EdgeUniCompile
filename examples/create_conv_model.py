#!/usr/bin/env python3.13
"""
Create a single-conv ONNX model with memory usage > 3MB.
"""

import onnx
from onnx import numpy_helper
import numpy as np


def create_single_conv_model(output_path="single_conv_model.onnx"):
    """Create a single-conv ONNX model with memory usage > 3MB."""

    # Create the ONNX model
    model = onnx.ModelProto()
    model.ir_version = 7
    model.opset_import.append(onnx.OperatorSetIdProto())
    model.opset_import[0].version = 13

    # Create graph
    graph = model.graph
    graph.name = "single_conv_graph"

    # Input tensor (NCHW format: 1x3x224x224 = ~600KB)
    input_tensor = onnx.ValueInfoProto()
    input_tensor.name = "input"
    input_tensor.type.tensor_type.elem_type = onnx.TensorProto.FLOAT
    input_tensor.type.tensor_type.shape.dim.extend([
        onnx.TensorShapeProto.Dimension(dim_value=1),    # Batch size
        onnx.TensorShapeProto.Dimension(dim_value=3),    # Channels
        onnx.TensorShapeProto.Dimension(dim_value=224),  # Height
        onnx.TensorShapeProto.Dimension(dim_value=224)   # Width
    ])
    graph.input.append(input_tensor)

    # Weights tensor (32 filters, 3x3 kernel: 32x3x3x3 = 864 bytes)
    weights = np.random.randn(32, 3, 3, 3).astype(np.float32)
    weights_tensor = numpy_helper.from_array(weights, "weights")
    graph.initializer.append(weights_tensor)

    # Bias tensor (32 elements: 128 bytes)
    bias = np.random.randn(32).astype(np.float32)
    bias_tensor = numpy_helper.from_array(bias, "bias")
    graph.initializer.append(bias_tensor)

    # Output tensor (NCHW format: 1x32x224x224 = ~6.4MB)
    output_tensor = onnx.ValueInfoProto()
    output_tensor.name = "output"
    output_tensor.type.tensor_type.elem_type = onnx.TensorProto.FLOAT
    output_tensor.type.tensor_type.shape.dim.extend([
        onnx.TensorShapeProto.Dimension(dim_value=1),    # Batch size
        onnx.TensorShapeProto.Dimension(dim_value=32),   # Channels
        onnx.TensorShapeProto.Dimension(dim_value=224),  # Height
        onnx.TensorShapeProto.Dimension(dim_value=224)   # Width
    ])
    graph.output.append(output_tensor)

    # Conv node
    conv_node = onnx.NodeProto()
    conv_node.name = "conv"
    conv_node.op_type = "Conv"
    conv_node.input.extend(["input", "weights", "bias"])
    conv_node.output.extend(["output"])

    # Conv attributes
    conv_node.attribute.append(onnx.helper.make_attribute("kernel_shape", [3, 3]))
    conv_node.attribute.append(onnx.helper.make_attribute("strides", [1, 1]))
    conv_node.attribute.append(onnx.helper.make_attribute("pads", [1, 1, 1, 1]))
    conv_node.attribute.append(onnx.helper.make_attribute("dilations", [1, 1]))
    conv_node.attribute.append(onnx.helper.make_attribute("group", 1))

    graph.node.append(conv_node)

    # Check model
    onnx.checker.check_model(model)

    # Save model
    onnx.save(model, output_path)
    print(f"Model saved to {output_path}")

    # Print model info
    print(f"Input shape: {input_tensor.type.tensor_type.shape}")
    print(f"Output shape: {output_tensor.type.tensor_type.shape}")
    print(f"Weights shape: {weights_tensor.dims}")
    print(f"Memory usage (inputs/outputs): ~{(1*3*224*224*4 + 1*32*224*224*4) / (1024*1024):.1f}MB")

    return model


if __name__ == "__main__":
    create_single_conv_model()
