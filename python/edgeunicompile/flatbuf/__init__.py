"""
FlatBuffer support for EdgeUniCompile.

This module provides functionality to:
1. Build FlatBuffers from EdgeUniCompile Graph
2. Parse FlatBuffers to EdgeUniCompile Graph
3. Generate FlatBuffer code from schema

The FlatBuffer schema is located at:
    include/edgeunicompile/flatbuf/edgeunicompile.fbs

To regenerate FlatBuffer code:
    flatc -p --gen-mutable include/edgeunicompile/flatbuf/edgeunicompile.fbs
    flatc -c --gen-mutable include/edgeunicompile/flatbuf/edgeunicompile.fbs
"""

import struct
from typing import Optional, BinaryIO
from flatbuffers import Builder as FBBuilder

from edgeunicompile.core import Context, Status
from edgeunicompile.ir import Graph, Node, Tensor

# Import generated FlatBuffer code (will be available after running flatc)
try:
    from edgeunicompile.flatbuf import Graph as FBGraph
    from edgeunicompile.flatbuf import Tensor as FBTensor
    from edgeunicompile.flatbuf import Node as FBNode
    from edgeunicompile.flatbuf import DataType as FBDataType
    from edgeunicompile.flatbuf import OpType as FBOpType
    from edgeunicompile.flatbuf import Shape as FBShape
    FLATBUFFER_GENERATED_AVAILABLE = True
except ImportError:
    FLATBUFFER_GENERATED_AVAILABLE = False


# Mapping between EdgeUniCompile DataType and FlatBuffer DataType
DTYPE_TO_FB = {
    "Unknown": 0,
    "Float32": 1,
    "Float16": 2,
    "Int32": 3,
    "Int16": 4,
    "Int8": 5,
    "UInt32": 6,
    "UInt16": 7,
    "UInt8": 8,
    "Bool": 9,
    "Complex64": 10,
}

FB_TO_DTYPE = {v: k for k, v in DTYPE_TO_FB.items()}

# Mapping between OpType strings and FlatBuffer OpType
OP_TYPE_TO_FB = {
    "Unknown": 0,
    "Add": 1,
    "Subtract": 2,
    "Multiply": 3,
    "Divide": 4,
    "Conv2D": 5,
    "MaxPool2D": 6,
    "AveragePool2D": 7,
    "Relu": 8,
    "Sigmoid": 9,
    "Tanh": 10,
    "Softmax": 11,
    "MatMul": 12,
    "Reshape": 13,
    "Transpose": 14,
    "Concat": 15,
    "Slice": 16,
    "Pad": 17,
    "Gemm": 18,
    "BatchNormalization": 19,
    "Dropout": 20,
    "Flatten": 21,
    "GlobalAveragePool": 22,
    "LRN": 23,
    "InstanceNormalization": 24,
    "LeakyRelu": 25,
    "Elu": 26,
    "Selu": 27,
    "HardSigmoid": 28,
    "DepthwiseConv2D": 30,
    "ConvTranspose2D": 31,
    "Eltwise": 75,  # Element-wise operations
}

FB_TO_OP_TYPE = {v: k for k, v in OP_TYPE_TO_FB.items()}


class FlatBufferBuilder:
    """
    Builder for creating FlatBuffers from EdgeUniCompile Graph.

    This class handles the conversion from EdgeUniCompile Graph IR to
    FlatBuffer format.

    If the generated FlatBuffer code is available (via running flatc on the schema),
    it will use the real FlatBuffer format. Otherwise, it falls back to JSON serialization.
    """

    @staticmethod
    def build(graph: Graph,
              context: Optional[Context] = None) -> bytes:
        """
        Build a FlatBuffer from the given graph.

        Args:
            graph: EdgeUniCompile Graph to serialize.
            context: Optional compilation context.

        Returns:
            Serialized FlatBuffer bytes.
        """
        if FLATBUFFER_GENERATED_AVAILABLE:
            return FlatBufferBuilder._build_native(graph, context)
        else:
            return FlatBufferBuilder._build_json(graph, context)

    @staticmethod
    def _build_native(graph: Graph, context: Optional[Context] = None) -> bytes:
        """Build using generated FlatBuffer code."""
        builder = FBBuilder(1024)

        # Create tensors
        tensor_offsets = []
        tensor_id_map = {}

        for i, tensor in enumerate(graph.tensors):
            tensor_id_map[tensor] = i

            # Create shape
            dims = builder.CreateVector(tensor.shape.dims)
            shape_offset = FBShape.CreateShape(builder, dims)

            # Create data vector
            data_offset = 0
            if tensor.data:
                data_offset = builder.CreateVector(tensor.data)

            # Create name string
            name_offset = builder.CreateString(tensor.name)

            # Create tensor
            tensor_offset = FBTensor.CreateTensor(
                builder,
                id=i,
                name=name_offset,
                dtype=DTYPE_TO_FB.get(str(tensor.dtype.value), 0),
                shape=shape_offset,
                data=data_offset,
                is_constant=tensor.is_constant() if hasattr(tensor, 'is_constant') else False
            )
            tensor_offsets.append(tensor_offset)

        # Create nodes
        node_offsets = []
        for node in graph.nodes:
            # Create input/output tensor ID vectors
            input_ids = [tensor_id_map.get(t, 0) for t in node.inputs]
            output_ids = [tensor_id_map.get(t, 0) for t in node.outputs]
            inputs_offset = builder.CreateVector(input_ids)
            outputs_offset = builder.CreateVector(output_ids)

            # Create name string
            name_offset = builder.CreateString(node.name)

            # Create node
            node_offset = FBNode.CreateNode(
                builder,
                name=name_offset,
                op_type=OP_TYPE_TO_FB.get(node.op_type, 0),
                inputs=inputs_offset,
                outputs=outputs_offset
            )
            node_offsets.append(node_offset)

        # Create input/output tensor ID lists
        input_tensor_ids = [tensor_id_map.get(t, 0) for t in graph.input_tensors]
        output_tensor_ids = [tensor_id_map.get(t, 0) for t in graph.output_tensors]
        input_ids_offset = builder.CreateVector(input_tensor_ids)
        output_ids_offset = builder.CreateVector(output_tensor_ids)

        # Create graph name
        graph_name_offset = builder.CreateString(graph.name)

        # Create node and tensor vectors
        nodes_offset = builder.CreateVector(node_offsets)
        tensors_offset = builder.CreateVector(tensor_offsets)

        # Create the Graph table
        graph_offset = FBGraph.CreateGraph(
            builder,
            version=builder.CreateString("0.1.0"),
            name=graph_name_offset,
            nodes=nodes_offset,
            tensors=tensors_offset,
            input_tensor_ids=input_ids_offset,
            output_tensor_ids=output_ids_offset
        )

        # Finish the buffer
        builder.Finish(graph_offset)

        return builder.Output()

    @staticmethod
    def _build_json(graph: Graph, context: Optional[Context] = None) -> bytes:
        """Build using JSON fallback (when flatc generated code is not available)."""
        import json

        data = {
            "version": "0.1.0",
            "name": graph.name,
            "nodes": [],
            "tensors": [],
            "input_tensors": [],
            "output_tensors": []
        }

        # Serialize tensors
        tensor_id_map = {}
        for i, tensor in enumerate(graph.tensors):
            tensor_data = {
                "id": i,
                "name": tensor.name,
                "data_type": str(tensor.dtype.value),
                "shape": tensor.shape.dims,
                "data": list(tensor.data) if tensor.data else []
            }
            data["tensors"].append(tensor_data)
            tensor_id_map[tensor] = i

        # Serialize nodes
        for node in graph.nodes:
            node_data = {
                "name": node.name,
                "op_type": node.op_type,
                "inputs": [tensor_id_map[tensor] for tensor in node.inputs],
                "outputs": [tensor_id_map[tensor] for tensor in node.outputs],
                "attributes": dict(node.attributes)
            }
            data["nodes"].append(node_data)

        # Serialize input/output tensors
        data["input_tensors"] = [tensor_id_map[tensor] for tensor in graph.input_tensors]
        data["output_tensors"] = [tensor_id_map[tensor] for tensor in graph.output_tensors]

        # Convert to JSON and then bytes
        json_str = json.dumps(data, indent=0)
        return bytes(json_str, "utf-8")

    @staticmethod
    def parse(data: bytes,
              context: Optional[Context] = None) -> Graph:
        """
        Parse a FlatBuffer into EdgeUniCompile Graph.

        Args:
            data: Serialized FlatBuffer bytes.
            context: Optional compilation context.

        Returns:
            EdgeUniCompile Graph.
        """
        if FLATBUFFER_GENERATED_AVAILABLE:
            return FlatBufferBuilder._parse_native(data, context)
        else:
            return FlatBufferBuilder._parse_json(data, context)

    @staticmethod
    def _parse_native(data: bytes, context: Optional[Context] = None) -> Graph:
        """Parse using generated FlatBuffer code."""
        # Get root as Graph
        fb_graph = FBGraph.Graph.GetRootAs(data)

        if context is None:
            from edgeunicompile.core import Context
            context = Context()

        graph = Graph(fb_graph.Name().decode('utf-8'))

        # Parse tensors
        id_to_tensor = {}
        for i in range(fb_graph.TensorsLength()):
            fb_tensor = fb_graph.Tensors(i)

            from edgeunicompile.core import Shape, DataType

            # Get shape
            shape_dims = []
            if fb_tensor.Shape():
                for j in range(fb_tensor.Shape().DimsLength()):
                    shape_dims.append(fb_tensor.Shape().Dims(j))

            # Create tensor
            tensor = Tensor(
                fb_tensor.Name().decode('utf-8'),
                DataType(FB_TO_DTYPE.get(fb_tensor.Dtype(), 1)),  # Default to Float32
                Shape(shape_dims)
            )

            # Get data
            if fb_tensor.DataIsNone() and fb_tensor.DataLength() > 0:
                tensor.data = bytes(fb_tensor.DataAsNumpy().tobytes())

            graph.add_tensor(tensor)
            id_to_tensor[i] = tensor

        # Parse nodes
        for i in range(fb_graph.NodesLength()):
            fb_node = fb_graph.Nodes(i)

            node = Node(
                fb_node.Name().decode('utf-8'),
                FB_TO_OP_TYPE.get(fb_node.OpType(), "Unknown")
            )

            # Add inputs
            for j in range(fb_node.InputsLength()):
                tensor_id = fb_node.Inputs(j)
                if tensor_id in id_to_tensor:
                    node.add_input(id_to_tensor[tensor_id])

            # Add outputs
            for j in range(fb_node.OutputsLength()):
                tensor_id = fb_node.Outputs(j)
                if tensor_id in id_to_tensor:
                    node.add_output(id_to_tensor[tensor_id])

            graph.add_node(node)

        # Add input tensors
        for i in range(fb_graph.InputTensorIdsLength()):
            tensor_id = fb_graph.InputTensorIds(i)
            if tensor_id in id_to_tensor:
                graph.add_input_tensor(id_to_tensor[tensor_id])

        # Add output tensors
        for i in range(fb_graph.OutputTensorIdsLength()):
            tensor_id = fb_graph.OutputTensorIds(i)
            if tensor_id in id_to_tensor:
                graph.add_output_tensor(id_to_tensor[tensor_id])

        return graph

    @staticmethod
    def _parse_json(data: bytes, context: Optional[Context] = None) -> Graph:
        """Parse JSON fallback (when flatc generated code is not available)."""
        import json

        json_str = data.decode("utf-8")
        data_dict = json.loads(json_str)

        if context is None:
            from edgeunicompile.core import Context
            context = Context()

        graph = Graph(data_dict["name"])

        # Create tensors
        tensor_list = []
        id_to_tensor = {}
        for tensor_data in data_dict["tensors"]:
            from edgeunicompile.core import Shape, DataType
            tensor = Tensor(
                tensor_data["name"],
                DataType(tensor_data["data_type"]),
                Shape(list(tensor_data["shape"]))
            )
            if tensor_data["data"]:
                tensor.data = bytes(tensor_data["data"])
            graph.add_tensor(tensor)
            tensor_list.append(tensor)
            id_to_tensor[tensor_data["id"]] = tensor

        # Create nodes
        for node_data in data_dict["nodes"]:
            node = Node(node_data["name"], node_data["op_type"])

            for tensor_id in node_data["inputs"]:
                node.add_input(id_to_tensor[tensor_id])

            for tensor_id in node_data["outputs"]:
                node.add_output(id_to_tensor[tensor_id])

            for key, value in node_data.get("attributes", {}).items():
                node.set_attribute(key, value)

            graph.add_node(node)

        # Add input tensors
        for tensor_id in data_dict["input_tensors"]:
            graph.add_input_tensor(id_to_tensor[tensor_id])

        # Add output tensors
        for tensor_id in data_dict["output_tensors"]:
            graph.add_output_tensor(id_to_tensor[tensor_id])

        return graph

    @staticmethod
    def save_to_file(graph: Graph,
                     filename: str,
                     context: Optional[Context] = None) -> Status:
        """
        Save a graph to a FlatBuffer file.

        Args:
            graph: EdgeUniCompile Graph to save.
            filename: Output file path.
            context: Optional compilation context.

        Returns:
            Status indicating success or failure.
        """
        try:
            data = FlatBufferBuilder.build(graph, context)
            with open(filename, "wb") as f:
                f.write(data)
            return Status()
        except Exception as e:
            from edgeunicompile.core import StatusCode
            return Status(code=StatusCode.ERROR, message=f"Failed to save to file: {str(e)}")

    @staticmethod
    def load_from_file(filename: str,
                       context: Optional[Context] = None) -> Graph:
        """
        Load a graph from a FlatBuffer file.

        Args:
            filename: Input file path.
            context: Optional compilation context.

        Returns:
            EdgeUniCompile Graph.

        Raises:
            FileNotFoundError: If file not found.
            IOError: If file read fails.
        """
        try:
            with open(filename, "rb") as f:
                data = f.read()
            return FlatBufferBuilder.parse(data, context)
        except FileNotFoundError:
            raise FileNotFoundError(f"File '{filename}' not found")
        except Exception as e:
            raise IOError(f"Failed to load file: {str(e)}")


class FlatBufferParser:
    """
    Parser for loading FlatBuffers.

    This is a legacy alias for FlatBufferBuilder.
    """
    parse = staticmethod(FlatBufferBuilder.parse)
    load_from_file = staticmethod(FlatBufferBuilder.load_from_file)


# Convenience functions
def build_flatbuffer(graph: Graph, context: Optional[Context] = None) -> bytes:
    """Convenience function to build FlatBuffer."""
    return FlatBufferBuilder.build(graph, context)


def parse_flatbuffer(data: bytes, context: Optional[Context] = None) -> Graph:
    """Convenience function to parse FlatBuffer."""
    return FlatBufferBuilder.parse(data, context)


def save_graph_to_flatbuffer(graph: Graph,
                             filename: str,
                             context: Optional[Context] = None):
    """Convenience function to save graph to file."""
    return FlatBufferBuilder.save_to_file(graph, filename, context)


def load_graph_from_flatbuffer(filename: str,
                               context: Optional[Context] = None) -> Graph:
    """Convenience function to load graph from file."""
    return FlatBufferBuilder.load_from_file(filename, context)


def generate_flatbuffer_code(output_dir: str = None):
    """
    Generate FlatBuffer code from the schema.

    This function runs flatc to generate Python code from the schema file.

    Args:
        output_dir: Output directory for generated code. If None, uses the
                    flatbuf package directory.
    """
    import subprocess
    import os

    schema_file = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
        "include", "edgeunicompile", "flatbuf", "edgeunicompile.fbs"
    )

    if output_dir is None:
        output_dir = os.path.dirname(__file__)

    if not os.path.exists(schema_file):
        raise FileNotFoundError(f"Schema file not found: {schema_file}")

    # Run flatc to generate Python code
    cmd = ["flatc", "-p", "--gen-mutable", "-o", output_dir, schema_file]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"flatc failed: {result.stderr}")

    print(f"Generated FlatBuffer Python code in: {output_dir}")


# Module level functions
__all__ = [
    "FlatBufferBuilder",
    "FlatBufferParser",
    "build_flatbuffer",
    "parse_flatbuffer",
    "save_graph_to_flatbuffer",
    "load_graph_from_flatbuffer",
    "generate_flatbuffer_code"
]
