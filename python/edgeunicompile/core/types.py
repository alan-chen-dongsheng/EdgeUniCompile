"""
Core type definitions and constants.
"""

from dataclasses import dataclass
from typing import List, Union, Any
from enum import Enum
from edgeunicompile.core import DataType


def get_data_type_size(dtype: DataType) -> int:
    """Get the size in bytes of a data type."""
    sizes = {
        DataType.FLOAT32: 4,
        DataType.FLOAT16: 2,
        DataType.INT32: 4,
        DataType.INT16: 2,
        DataType.INT8: 1,
        DataType.UINT32: 4,
        DataType.UINT16: 2,
        DataType.UINT8: 1,
        DataType.BOOL: 1,
        DataType.COMPLEX64: 8,
    }
    return sizes.get(dtype, 0)


def data_type_to_string(dtype: DataType) -> str:
    """Convert data type to string."""
    return dtype.value


def string_to_data_type(s: str) -> DataType:
    """Convert string to data type."""
    for dtype in DataType:
        if dtype.value == s.lower():
            return dtype
        if dtype.name.lower() == s.lower():
            return dtype
    raise ValueError(f"Unknown data type: {s}")


def op_type_to_string(op: str) -> str:
    """Convert operator type to string."""
    from edgeunicompile.core import OpType

    if isinstance(op, OpType):
        return op.value
    return str(op)


def string_to_op_type(s: str) -> str:
    """Convert string to operator type."""
    from edgeunicompile.core import OpType

    for op in OpType:
        if op.value == s:
            return op
        if op.name.lower() == s.lower():
            return op
    raise ValueError(f"Unknown operator type: {s}")


@dataclass
class AttributeValue:
    """Wrapper for various attribute value types."""
    value: Union[
        bool,
        int,
        float,
        str,
        List[int],
        List[float],
        List[str],
        List[List[int]],
        "Shape",
    ]

    def __post_init__(self):
        from edgeunicompile.core import Shape

        if isinstance(self.value, list) and all(isinstance(x, (list, tuple)) for x in self.value):
            if not all(isinstance(x, int) for sublist in self.value for x in sublist):
                raise ValueError("All elements in 2D list must be integers")
            self.value = Shape([x for sublist in self.value for x in sublist])

    def is_numeric(self) -> bool:
        """Check if value is numeric (int, float)."""
        return isinstance(self.value, (int, float))

    def is_list(self) -> bool:
        """Check if value is a list."""
        return isinstance(self.value, list)

    def is_string(self) -> bool:
        """Check if value is a string."""
        return isinstance(self.value, str)

    def is_boolean(self) -> bool:
        """Check if value is boolean."""
        return isinstance(self.value, bool)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        from edgeunicompile.core import Shape

        if isinstance(self.value, Shape):
            return {"type": "shape", "value": self.value.dims}
        return {"type": type(self.value).__name__, "value": self.value}

    @classmethod
    def from_dict(cls, data: dict) -> "AttributeValue":
        """Create from dictionary."""
        from edgeunicompile.core import Shape

        if "type" in data and data["type"] == "shape":
            return cls(Shape(data["value"]))
        return cls(data["value"])


def shape_from_dict(data: dict) -> "Shape":
    """Create shape from dictionary."""
    from edgeunicompile.core import Shape

    if "dims" in data:
        return Shape(data["dims"])
    elif isinstance(data, list):
        return Shape(data)
    raise ValueError("Invalid shape format")


def shape_to_dict(shape: "Shape") -> dict:
    """Convert shape to dictionary."""
    return {"dims": shape.dims}


# Common attribute names
ATTR_KERNEL_SIZE = "kernel_size"
ATTR_STRIDES = "strides"
ATTR_PADDING = "padding"
ATTR_DILATIONS = "dilations"
ATTR_GROUPS = "groups"
ATTR_AXIS = "axis"
ATTR_ORDER = "order"
ATTR_OUTPUT_SHAPE = "output_shape"
