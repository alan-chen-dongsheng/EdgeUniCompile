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
        # For now, create a simple flatbuffer structure
        # Build a placeholder buffer with graph metadata
        metadata = {
            "version": "0.1.0",
            "name": graph.name,
            "nodes": len(graph.nodes),
            "tensors": len(graph.tensors)
        }

        # Simple serialization for now (will be replaced with real FlatBuffer)
        data = bytearray()
        data.extend(struct.pack("<I", len(metadata["version"])))
        data.extend(metadata["version"].encode("utf-8"))

        data.extend(struct.pack("<I", len(metadata["name"])))
        data.extend(metadata["name"].encode("utf-8"))

        data.extend(struct.pack("<Q", metadata["nodes"]))
        data.extend(struct.pack("<Q", metadata["tensors"]))

        return bytes(data)

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
        # For now, create a placeholder graph
        # In real implementation, this will parse the actual FlatBuffer

        # Create a simple graph
        if context is None:
            from edgeunicompile.core import Context
            context = Context()

        graph = Graph("flatbuffer_graph")

        # Add some dummy nodes and tensors
        tensor1 = Tensor("input1", "float32", (2, 3))
        tensor2 = Tensor("input2", "float32", (2, 3))
        tensor3 = Tensor("output1", "float32", (2, 3))

        graph.add_tensor(tensor1)
        graph.add_tensor(tensor2)
        graph.add_tensor(tensor3)
        graph.add_input_tensor(tensor1)
        graph.add_input_tensor(tensor2)
        graph.add_output_tensor(tensor3)

        node = Node("add_node", "Add")
        node.add_input(tensor1)
        node.add_input(tensor2)
        node.add_output(tensor3)
        graph.add_node(node)

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
            return Status(code=1, message=f"Failed to save to file: {str(e)}")

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
