"""
FlatBuffer support for EdgeUniCompile.

This module provides functionality to:
1. Build FlatBuffers from EdgeUniCompile Graph
2. Parse FlatBuffers to EdgeUniCompile Graph
3. Generate FlatBuffer schema
"""

import struct
from typing import Optional, BinaryIO
from flatbuffers import Builder as FBBuilder

from edgeunicompile.core import Context, Status
from edgeunicompile.ir import Graph, Node, Tensor


class FlatBufferBuilder:
    """
    Builder for creating FlatBuffers from EdgeUniCompile Graph.

    This class handles the conversion from EdgeUniCompile Graph IR to
    FlatBuffer format.
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
        import json

        # For now, serialize to JSON and then encode as bytes
        # This will be replaced with real FlatBuffer in the future
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
                "data_type": tensor.dtype.value,
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
        import json

        # For now, parse JSON from bytes
        # This will be replaced with real FlatBuffer in the future
        json_str = data.decode("utf-8")
        data = json.loads(json_str)

        if context is None:
            from edgeunicompile.core import Context
            context = Context()

        graph = Graph(data["name"])

        # Create tensors
        tensor_list = []
        id_to_tensor = {}
        for tensor_data in data["tensors"]:
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
        for node_data in data["nodes"]:
            node = Node(node_data["name"], node_data["op_type"])

            for tensor_id in node_data["inputs"]:
                node.add_input(id_to_tensor[tensor_id])

            for tensor_id in node_data["outputs"]:
                node.add_output(id_to_tensor[tensor_id])

            for key, value in node_data.get("attributes", {}).items():
                node.set_attribute(key, value)

            graph.add_node(node)

        # Add input tensors
        for tensor_id in data["input_tensors"]:
            graph.add_input_tensor(id_to_tensor[tensor_id])

        # Add output tensors
        for tensor_id in data["output_tensors"]:
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


# Module level functions
__all__ = [
    "FlatBufferBuilder",
    "FlatBufferParser",
    "build_flatbuffer",
    "parse_flatbuffer",
    "save_graph_to_flatbuffer",
    "load_graph_from_flatbuffer"
]
