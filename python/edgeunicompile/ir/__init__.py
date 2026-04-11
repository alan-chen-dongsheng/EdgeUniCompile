"""
IR module containing the graph, node, and tensor classes for representing
AI models in EdgeUniCompile.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from edgeunicompile.core import Shape, DataType, Status, StatusCode
from edgeunicompile.core.types import ATTR_KERNEL_SIZE, ATTR_STRIDES, ATTR_PADDING


@dataclass
class Tensor:
    """
    Tensor representation in the graph.

    A tensor is a multi-dimensional array with:
        - name: Unique identifier
        - dtype: Data type
        - shape: Shape of the tensor
        - data: Optional data (for constants)
        - producer_node: Optional node that produces this tensor
    """
    name: str
    dtype: DataType
    shape: Shape
    data: Optional[bytes] = None
    producer_node: Optional[str] = None

    def __post_init__(self):
        """Convert dtype and shape to proper types if needed."""
        from edgeunicompile.core import Shape, DataType

        # Convert dtype from string to DataType enum if needed
        if isinstance(self.dtype, str):
            dtype_value = self.dtype.lower()
            for dtype in DataType:
                if dtype.value == dtype_value or dtype.name.lower() == dtype_value:
                    self.dtype = dtype
                    break

        # Convert shape from list/tuple to Shape if needed
        if not isinstance(self.shape, Shape):
            self.shape = Shape(list(self.shape))

    def num_elements(self) -> int:
        """Calculate total number of elements in the tensor."""
        return self.shape.num_elements()

    def element_size(self) -> int:
        """Calculate size of each element in bytes."""
        from edgeunicompile.core.types import get_data_type_size
        return get_data_type_size(self.dtype)

    def total_size(self) -> int:
        """Calculate total size in bytes."""
        return self.num_elements() * self.element_size()

    def is_constant(self) -> bool:
        """Check if tensor is a constant (has data)."""
        return self.data is not None

    def __hash__(self):
        return hash((self.name, self.dtype, tuple(self.shape.dims)))

    def __eq__(self, other):
        if not isinstance(other, Tensor):
            return False
        return (
            self.name == other.name and
            self.dtype == other.dtype and
            self.shape == other.shape and
            self.data == other.data and
            self.producer_node == other.producer_node
        )

    def __str__(self):
        data_info = "(data)" if self.data else ""
        return f"{self.name}: {self.dtype} {self.shape} {data_info}"


@dataclass
class Node:
    """
    Node representation in the graph.

    A node represents an operation with:
        - name: Unique identifier
        - op_type: Operation type (e.g., Conv2D, Relu)
        - inputs: Input tensors
        - outputs: Output tensors
        - attributes: Operation attributes (e.g., kernel size, strides)
    """
    name: str
    op_type: str
    inputs: List[Tensor] = field(default_factory=list)
    outputs: List[Tensor] = field(default_factory=list)
    attributes: Dict[str, Any] = field(default_factory=dict)

    def add_input(self, tensor: Tensor):
        """Add an input tensor."""
        if tensor not in self.inputs:
            self.inputs.append(tensor)

    def remove_input(self, tensor: Tensor):
        """Remove an input tensor."""
        if tensor in self.inputs:
            self.inputs.remove(tensor)

    def add_output(self, tensor: Tensor):
        """Add an output tensor."""
        if tensor not in self.outputs:
            self.outputs.append(tensor)

    def remove_output(self, tensor: Tensor):
        """Remove an output tensor."""
        if tensor in self.outputs:
            self.outputs.remove(tensor)

    def set_attribute(self, key: str, value: Any):
        """Set an attribute."""
        self.attributes[key] = value

    def get_attribute(self, key: str, default: Any = None) -> Any:
        """Get an attribute value."""
        return self.attributes.get(key, default)

    def has_attribute(self, key: str) -> bool:
        """Check if an attribute exists."""
        return key in self.attributes

    def __hash__(self):
        return hash((self.name, self.op_type))

    def __eq__(self, other):
        if not isinstance(other, Node):
            return False
        return (
            self.name == other.name and
            self.op_type == other.op_type and
            self.inputs == other.inputs and
            self.outputs == other.outputs and
            self.attributes == other.attributes
        )

    def __str__(self):
        input_names = ", ".join(t.name for t in self.inputs)
        output_names = ", ".join(t.name for t in self.outputs)
        return f"{self.name}: {self.op_type}({input_names}) -> {output_names}"


@dataclass
class Graph:
    """
    Graph representation of an AI model.

    A graph contains:
        - name: Name of the graph
        - nodes: List of operations
        - tensors: List of tensors
        - input_tensors: Input tensors
        - output_tensors: Output tensors
    """
    name: str
    nodes: List[Node] = field(default_factory=list)
    tensors: List[Tensor] = field(default_factory=list)
    input_tensors: List[Tensor] = field(default_factory=list)
    output_tensors: List[Tensor] = field(default_factory=list)

    # Internal maps for quick lookups
    _node_map: Dict[str, Node] = field(default_factory=dict)
    _tensor_map: Dict[str, Tensor] = field(default_factory=dict)

    def __post_init__(self):
        # Build internal maps
        for node in self.nodes:
            self._node_map[node.name] = node
        for tensor in self.tensors:
            self._tensor_map[tensor.name] = tensor

    def add_node(self, node: Node):
        """Add a node to the graph."""
        if node.name in self._node_map:
            raise ValueError(f"Node '{node.name}' already exists in graph")
        self.nodes.append(node)
        self._node_map[node.name] = node

    def remove_node(self, node: Node):
        """Remove a node from the graph."""
        if node.name in self._node_map:
            self.nodes.remove(node)
            del self._node_map[node.name]

    def get_node(self, name: str) -> Optional[Node]:
        """Get a node by name."""
        return self._node_map.get(name)

    def has_node(self, name: str) -> bool:
        """Check if a node exists by name."""
        return name in self._node_map

    def add_tensor(self, tensor: Tensor):
        """Add a tensor to the graph."""
        if tensor.name in self._tensor_map:
            raise ValueError(f"Tensor '{tensor.name}' already exists in graph")
        self.tensors.append(tensor)
        self._tensor_map[tensor.name] = tensor

    def remove_tensor(self, tensor: Tensor):
        """Remove a tensor from the graph."""
        if tensor.name in self._tensor_map:
            self.tensors.remove(tensor)
            del self._tensor_map[tensor.name]

    def get_tensor(self, name: str) -> Optional[Tensor]:
        """Get a tensor by name."""
        return self._tensor_map.get(name)

    def has_tensor(self, name: str) -> bool:
        """Check if a tensor exists by name."""
        return name in self._tensor_map

    def add_input_tensor(self, tensor: Tensor):
        """Add an input tensor to the graph."""
        if tensor not in self.input_tensors:
            self.input_tensors.append(tensor)
        if tensor not in self.tensors:
            self.add_tensor(tensor)

    def remove_input_tensor(self, tensor: Tensor):
        """Remove an input tensor from the graph."""
        if tensor in self.input_tensors:
            self.input_tensors.remove(tensor)

    def add_output_tensor(self, tensor: Tensor):
        """Add an output tensor to the graph."""
        if tensor not in self.output_tensors:
            self.output_tensors.append(tensor)
        if tensor not in self.tensors:
            self.add_tensor(tensor)

    def remove_output_tensor(self, tensor: Tensor):
        """Remove an output tensor from the graph."""
        if tensor in self.output_tensors:
            self.output_tensors.remove(tensor)

    def get_topological_order(self) -> List[Node]:
        """
        Get nodes in topological order.

        Returns:
            List of nodes in topological order
        """
        from collections import deque

        # Build adjacency list and in-degree count
        adj = {}
        in_degree = {}

        for node in self.nodes:
            adj[node.name] = []
            in_degree[node.name] = 0

        for node in self.nodes:
            for output in node.outputs:
                for other_node in self.nodes:
                    if node != other_node and any(t == output for t in other_node.inputs):
                        adj[node.name].append(other_node.name)
                        in_degree[other_node.name] += 1

        # Kahn's algorithm for topological sort
        queue = deque(name for name, degree in in_degree.items() if degree == 0)
        topological = []

        while queue:
            u = queue.popleft()
            topological.append(self._node_map[u])

            for v in adj[u]:
                in_degree[v] -= 1
                if in_degree[v] == 0:
                    queue.append(v)

        return topological

    def is_valid(self) -> Status:
        """
        Check if the graph is valid.

        Returns:
            Status indicating validity
        """
        # Check nodes
        for node in self.nodes:
            if not node.name or len(node.name.strip()) == 0:
                return Status(StatusCode.INVALID_ARGUMENT, "Node has no name")

            if len(node.inputs) > len(set(node.inputs)):
                return Status(StatusCode.INVALID_ARGUMENT, f"Node '{node.name}' has duplicate inputs")

            if len(node.outputs) > len(set(node.outputs)):
                return Status(StatusCode.INVALID_ARGUMENT, f"Node '{node.name}' has duplicate outputs")

        # Check tensors
        for tensor in self.tensors:
            if not tensor.name or len(tensor.name.strip()) == 0:
                return Status(StatusCode.INVALID_ARGUMENT, "Tensor has no name")

            if not tensor.shape.is_valid():
                return Status(StatusCode.INVALID_ARGUMENT,
                            f"Tensor '{tensor.name}' has invalid shape: {tensor.shape}")

        # Check inputs
        for tensor in self.input_tensors:
            if tensor not in self.tensors:
                return Status(StatusCode.INVALID_ARGUMENT,
                            f"Input tensor '{tensor.name}' is not in graph")

        # Check outputs
        for tensor in self.output_tensors:
            if tensor not in self.tensors:
                return Status(StatusCode.INVALID_ARGUMENT,
                            f"Output tensor '{tensor.name}' is not in graph")

        return Status.ok()

    def __str__(self):
        header = f"Graph: {self.name}\n"
        inputs = f"Inputs: [{', '.join(t.name for t in self.input_tensors)}]\n"
        outputs = f"Outputs: [{', '.join(t.name for t in self.output_tensors)}]\n"

        nodes = "Nodes:\n"
        for node in self.nodes:
            nodes += f"  {str(node)}\n"

        tensors = "Tensors:\n"
        for tensor in self.tensors:
            tensors += f"  {str(tensor)}\n"

        return header + inputs + outputs + nodes + tensors
