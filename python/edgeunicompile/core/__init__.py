"""
Core module containing basic types and context management.
"""

from dataclasses import dataclass
from typing import List, Optional, Union, Any
from enum import Enum
import json


class DataType(Enum):
    FLOAT32 = "float32"
    FLOAT16 = "float16"
    INT32 = "int32"
    INT16 = "int16"
    INT8 = "int8"
    UINT32 = "uint32"
    UINT16 = "uint16"
    UINT8 = "uint8"
    BOOL = "bool"
    COMPLEX64 = "complex64"


class OpType(Enum):
    ADD = "Add"
    SUBTRACT = "Subtract"
    MULTIPLY = "Multiply"
    DIVIDE = "Divide"
    CONV2D = "Conv2D"
    MAXPOOL2D = "MaxPool2D"
    AVERAGEPOOL2D = "AveragePool2D"
    RELU = "Relu"
    SIGMOID = "Sigmoid"
    TANH = "Tanh"
    SOFTMAX = "Softmax"
    MATMUL = "MatMul"
    RESHAPE = "Reshape"
    TRANSPOSE = "Transpose"


class StatusCode(Enum):
    OK = 0
    ERROR = 1
    INVALID_ARGUMENT = 2
    NOT_FOUND = 3
    NOT_IMPLEMENTED = 4
    INTERNAL = 5
    RESOURCE_EXHAUSTED = 6


@dataclass
class Shape:
    """Shape of a tensor (N-dimensional array)."""
    dims: List[int]

    def num_elements(self) -> int:
        """Calculate total number of elements in the tensor."""
        num = 1
        for dim in self.dims:
            num *= dim
        return num

    def rank(self) -> int:
        """Get the rank (number of dimensions) of the tensor."""
        return len(self.dims)

    def is_scalar(self) -> bool:
        """Check if the shape represents a scalar (no dimensions)."""
        return len(self.dims) == 0

    def is_valid(self) -> bool:
        """Check if the shape is valid (all dimensions positive)."""
        return all(dim > 0 for dim in self.dims)

    def to_string(self) -> str:
        """Convert shape to string representation."""
        return f"[{', '.join(str(d) for d in self.dims)}]"

    def __str__(self):
        return self.to_string()

    def __eq__(self, other):
        if isinstance(other, Shape):
            return self.dims == other.dims
        return False

    def __hash__(self):
        return hash(tuple(self.dims))


@dataclass
class Status:
    """Status object for error handling."""
    code: StatusCode = StatusCode.OK
    message: str = ""

    @classmethod
    def ok(cls):
        """Create an OK status."""
        return cls()

    @classmethod
    def error(cls, message: str):
        """Create an error status."""
        return cls(StatusCode.ERROR, message)

    @classmethod
    def invalid_argument(cls, message: str):
        """Create an invalid argument status."""
        return cls(StatusCode.INVALID_ARGUMENT, message)

    @classmethod
    def not_found(cls, message: str):
        """Create a not found status."""
        return cls(StatusCode.NOT_FOUND, message)

    @classmethod
    def not_implemented(cls, message: str):
        """Create a not implemented status."""
        return cls(StatusCode.NOT_IMPLEMENTED, message)

    @classmethod
    def internal(cls, message: str):
        """Create an internal error status."""
        return cls(StatusCode.INTERNAL, message)

    @classmethod
    def resource_exhausted(cls, message: str):
        """Create a resource exhausted status."""
        return cls(StatusCode.RESOURCE_EXHAUSTED, message)

    def is_ok(self) -> bool:
        """Check if status is OK."""
        return self.code == StatusCode.OK

    def is_error(self) -> bool:
        """Check if status is an error."""
        return self.code != StatusCode.OK

    def to_string(self) -> str:
        """Convert status to string representation."""
        if self.code == StatusCode.OK:
            return "OK"
        return f"{self.code.name}: {self.message}"

    def __str__(self):
        return self.to_string()

    def __bool__(self):
        return self.is_ok()


@dataclass
class Context:
    """
    Context for the compiler with configuration and state.

    Configuration options:
        - opt_level: Optimization level (0-4)
        - sram_size: SRAM size in bytes
        - workload_size: Workload size in bytes
        - target_arch: Target architecture (e.g., "armv8")
        - debug_mode: Enable debug mode
        - verbose_mode: Enable verbose output
    """
    opt_level: int = 3
    sram_size: int = 32 * 1024 * 1024  # 32MB default
    workload_size: int = 0
    target_arch: str = "armv8"
    debug_mode: bool = False
    verbose_mode: bool = False

    _counters: dict = None

    def __post_init__(self):
        self._counters = {}

    def set_attribute(self, key: str, value: Any):
        """Set a custom attribute."""
        if key.startswith("__"):
            raise ValueError("Attribute keys cannot start with __")
        setattr(self, key, value)

    def get_attribute(self, key: str, default: Any = None) -> Any:
        """Get a custom attribute."""
        if hasattr(self, key):
            return getattr(self, key, default)
        return default

    def increment_counter(self, name: str, value: int = 1):
        """Increment a performance counter."""
        if name not in self._counters:
            self._counters[name] = 0
        self._counters[name] += value

    def get_counter(self, name: str, default: int = 0) -> int:
        """Get a performance counter value."""
        return self._counters.get(name, default)

    def get_all_counters(self) -> dict:
        """Get all performance counters."""
        return dict(self._counters)

    def to_dict(self) -> dict:
        """Convert context to dictionary."""
        result = {
            "opt_level": self.opt_level,
            "sram_size": self.sram_size,
            "workload_size": self.workload_size,
            "target_arch": self.target_arch,
            "debug_mode": self.debug_mode,
            "verbose_mode": self.verbose_mode,
            "counters": self._counters
        }
        # Include custom attributes
        for attr in dir(self):
            if not attr.startswith("__") and attr not in result and not callable(getattr(self, attr)):
                result[attr] = getattr(self, attr)
        return result

    def to_json(self) -> str:
        """Convert context to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: dict) -> "Context":
        """Create context from dictionary."""
        ctx = cls()
        for key, value in data.items():
            if key == "counters":
                ctx._counters = value
            elif hasattr(ctx, key):
                setattr(ctx, key, value)
            else:
                ctx.set_attribute(key, value)
        return ctx

    @classmethod
    def from_json(cls, json_str: str) -> "Context":
        """Create context from JSON string."""
        return cls.from_dict(json.loads(json_str))

    def __str__(self):
        return self.to_json()


def create_context(**kwargs) -> Context:
    """Create a new context with optional configuration."""
    ctx = Context()
    for key, value in kwargs.items():
        if hasattr(ctx, key):
            setattr(ctx, key, value)
        else:
            ctx.set_attribute(key, value)
    return ctx
