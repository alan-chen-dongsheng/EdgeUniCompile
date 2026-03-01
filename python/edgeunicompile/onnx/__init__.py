"""
ONNX model converter for EdgeUniCompile.

This module provides functionality to convert ONNX models to EdgeUniCompile
Graph representation.
"""

from typing import Optional
import onnx
import numpy as np
from edgeunicompile.core import Context, Status, Shape, DataType
from edgeunicompile.ir import Graph, Node, Tensor


class ONNXConverter:
    """
    Converter for ONNX models to EdgeUniCompile Graph.

    This class provides static methods to convert ONNX models to the
    internal EdgeUniCompile representation.
    """

    @staticmethod
    def convert(onnx_model: str or onnx.ModelProto,
                context: Optional[Context] = None) -> Graph:
        """
        Convert an ONNX model to EdgeUniCompile Graph.

        Args:
            onnx_model: Path to ONNX file or ONNX ModelProto object.
            context: Optional compilation context.

        Returns:
            EdgeUniCompile Graph.

        Raises:
            FileNotFoundError: If file not found.
            ValueError: If model is invalid.
        """
        # Load ONNX model if provided as file path
        if isinstance(onnx_model, str):
            onnx_model = onnx.load(onnx_model)

        # Create context if not provided
        if context is None:
            from edgeunicompile.core import Context
            context = Context()

        # Create graph
        graph = Graph(onnx_model.graph.name or "onnx_graph")

        # Convert tensors
        tensor_map = {}
        for tensor_proto in onnx_model.graph.initializer:
            tensor = ONNXConverter._convert_tensor(tensor_proto, context)
            graph.add_tensor(tensor)
            graph.add_input_tensor(tensor)
            tensor_map[tensor_proto.name] = tensor

        # Convert inputs
        for input_proto in onnx_model.graph.input:
            if input_proto.name not in tensor_map:
                tensor = ONNXConverter._convert_value_info(input_proto, context)
                graph.add_tensor(tensor)
                graph.add_input_tensor(tensor)
                tensor_map[input_proto.name] = tensor

        # Convert outputs
        for output_proto in onnx_model.graph.output:
            tensor = ONNXConverter._convert_value_info(output_proto, context)
            graph.add_tensor(tensor)
            graph.add_output_tensor(tensor)
            tensor_map[output_proto.name] = tensor

        # Convert nodes
        for node in onnx_model.graph.node:
            edge_node = ONNXConverter._convert_node(node, context, tensor_map)
            graph.add_node(edge_node)

        return graph

    @staticmethod
    def _convert_tensor(tensor_proto: onnx.TensorProto, context: Context) -> Tensor:
        """
        Convert ONNX TensorProto to EdgeUniCompile Tensor.

        Args:
            tensor_proto: ONNX TensorProto.
            context: Compilation context.

        Returns:
            EdgeUniCompile Tensor.
        """
        # Convert data type
        dtype = ONNXConverter._convert_data_type(tensor_proto.data_type)

        # Convert shape
        shape = Shape([dim.dim_value for dim in tensor_proto.dims])

        # Create tensor
        tensor = Tensor(tensor_proto.name, dtype, shape)

        # Convert data
        if tensor_proto.raw_data:
            tensor.data = tensor_proto.raw_data

        tensor.set_attribute("onnx_type", "tensor")
        return tensor

    @staticmethod
    def _convert_value_info(value_info: onnx.ValueInfoProto, context: Context) -> Tensor:
        """
        Convert ONNX ValueInfoProto to EdgeUniCompile Tensor.

        Args:
            value_info: ONNX ValueInfoProto.
            context: Compilation context.

        Returns:
            EdgeUniCompile Tensor.
        """
        # Convert data type
        dtype = ONNXConverter._convert_data_type(value_info.type.tensor_type.elem_type)

        # Convert shape
        shape = Shape([])
        if value_info.type.tensor_type.HasField("shape"):
            dims = []
            for dim in value_info.type.tensor_type.shape.dim:
                if dim.HasField("dim_value"):
                    dims.append(dim.dim_value)
                else:
                    dims.append(0)  # Unknown dimension
            shape = Shape(dims)

        # Create tensor
        tensor = Tensor(value_info.name, dtype, shape)

        # Add ONNX type info
        tensor.set_attribute("onnx_type", "value_info")
        return tensor

    @staticmethod
    def _convert_node(node_proto: onnx.NodeProto,
                      context: Context,
                      tensor_map: dict) -> Node:
        """
        Convert ONNX NodeProto to EdgeUniCompile Node.

        Args:
            node_proto: ONNX NodeProto.
            context: Compilation context.
            tensor_map: Mapping from ONNX tensor names to EdgeUniCompile tensors.

        Returns:
            EdgeUniCompile Node.
        """
        # Convert op type
        op_type = ONNXConverter._convert_op_type(node_proto.op_type)

        # Create node
        node = Node(node_proto.name or f"{node_proto.op_type}_{id(node_proto)}",
                    op_type)

        # Add inputs
        for input_name in node_proto.input:
            if input_name in tensor_map:
                node.add_input(tensor_map[input_name])

        # Add outputs
        for output_name in node_proto.output:
            if output_name in tensor_map:
                node.add_output(tensor_map[output_name])

        # Convert attributes
        node = ONNXConverter._convert_attributes(node, node_proto.attribute)

        return node

    @staticmethod
    def _convert_data_type(onnx_dtype: int) -> DataType:
        """
        Convert ONNX data type to EdgeUniCompile DataType.

        Args:
            onnx_dtype: ONNX data type identifier.

        Returns:
            EdgeUniCompile DataType.
        """
        type_map = {
            onnx.TensorProto.FLOAT: DataType.FLOAT32,
            onnx.TensorProto.FLOAT16: DataType.FLOAT16,
            onnx.TensorProto.INT32: DataType.INT32,
            onnx.TensorProto.INT16: DataType.INT16,
            onnx.TensorProto.INT8: DataType.INT8,
            onnx.TensorProto.UINT32: DataType.UINT32,
            onnx.TensorProto.UINT16: DataType.UINT16,
            onnx.TensorProto.UINT8: DataType.UINT8,
            onnx.TensorProto.BOOL: DataType.BOOL,
            onnx.TensorProto.COMPLEX64: DataType.COMPLEX64
        }

        return type_map.get(onnx_dtype, DataType.FLOAT32)

    @staticmethod
    def _convert_op_type(onnx_op_type: str) -> str:
        """
        Convert ONNX operator type to EdgeUniCompile operator type.

        Args:
            onnx_op_type: ONNX operator type.

        Returns:
            EdgeUniCompile operator type.
        """
        # Remove domain prefix
        if '.' in onnx_op_type:
            onnx_op_type = onnx_op_type.split('.')[-1]

        # Mapping of ONNX op types to EdgeUniCompile op types
        op_map = {
            "Add": "Add",
            "Sub": "Subtract",
            "Mul": "Multiply",
            "Div": "Divide",
            "Conv": "Conv2D",
            "MaxPool": "MaxPool2D",
            "AveragePool": "AveragePool2D",
            "Relu": "Relu",
            "Sigmoid": "Sigmoid",
            "Tanh": "Tanh",
            "Softmax": "Softmax",
            "MatMul": "MatMul",
            "Reshape": "Reshape",
            "Transpose": "Transpose"
        }

        return op_map.get(onnx_op_type, "Unknown")

    @staticmethod
    def _convert_attributes(node: Node, onnx_attributes: list) -> Node:
        """
        Convert ONNX attributes to EdgeUniCompile node attributes.

        Args:
            node: EdgeUniCompile Node.
            onnx_attributes: List of ONNX AttributeProto.

        Returns:
            Updated EdgeUniCompile Node.
        """
        for attr in onnx_attributes:
            value = ONNXConverter._convert_attribute_value(attr)
            node.set_attribute(attr.name, value)
        return node

    @staticmethod
    def _convert_attribute_value(attr) -> any:
        """
        Convert ONNX attribute value to Python object.

        Args:
            attr: ONNX AttributeProto.

        Returns:
            Python value.
        """
        if attr.HasField("i"):
            return attr.i
        elif attr.HasField("f"):
            return attr.f
        elif attr.HasField("s"):
            return attr.s.decode("utf-8")
        elif attr.HasField("t"):
            return ONNXConverter._convert_tensor(attr.t, None)
        elif attr.HasField("g"):
            return ONNXConverter._convert_graph(attr.g)
        elif attr.ints:
            return list(attr.ints)
        elif attr.floats:
            return list(attr.floats)
        elif attr.strings:
            return [s.decode("utf-8") for s in attr.strings]
        return None

    @staticmethod
    def _convert_graph(graph_proto: onnx.GraphProto) -> "Graph":
        """
        Convert ONNX subgraph to EdgeUniCompile Graph.

        Args:
            graph_proto: ONNX GraphProto.

        Returns:
            EdgeUniCompile Graph.
        """
        # Create a new graph
        graph = Graph(graph_proto.name or "subgraph")

        # For now, we don't convert subgraphs fully
        # This should be implemented if subgraph support is needed
        return graph

    @staticmethod
    def validate_model(onnx_model: str or onnx.ModelProto) -> Status:
        """
        Validate an ONNX model.

        Args:
            onnx_model: Path to ONNX file or ONNX ModelProto object.

        Returns:
            Status indicating validity.
        """
        if isinstance(onnx_model, str):
            try:
                onnx_model = onnx.load(onnx_model)
            except FileNotFoundError:
                return Status(error="Model file not found")
            except Exception as e:
                return Status(error=f"Failed to load model: {str(e)}")

        try:
            onnx.checker.check_model(onnx_model)
            return Status.ok()
        except Exception as e:
            return Status(error=f"Model validation failed: {str(e)}")

    @staticmethod
    def summarize_model(onnx_model: str or onnx.ModelProto) -> dict:
        """
        Get a summary of the ONNX model.

        Args:
            onnx_model: Path to ONNX file or ONNX ModelProto object.

        Returns:
            Summary dictionary.
        """
        if isinstance(onnx_model, str):
            onnx_model = onnx.load(onnx_model)

        return {
            "name": onnx_model.graph.name or "unknown",
            "opset_imports": [str(imp) for imp in onnx_model.opset_import],
            "input_count": len(onnx_model.graph.input),
            "output_count": len(onnx_model.graph.output),
            "node_count": len(onnx_model.graph.node),
            "tensor_count": len(onnx_model.graph.initializer)
        }
