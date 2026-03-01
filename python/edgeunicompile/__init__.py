"""
EdgeUniCompile - Edge Universal Compile Framework

A compiler framework for edge AI devices that supports:
- ONNX model to FlatBuffer conversion
- C++/Python mixed pass system
- SRAM-aware tiling
- MLIR integration

Copyright (c) 2024
"""

__version__ = "0.1.0"

from edgeunicompile.core import Context
from edgeunicompile.ir import Graph, Node, Tensor
from edgeunicompile.passes import PassManager, PassBase
from edgeunicompile.onnx import ONNXConverter
from edgeunicompile.flatbuf import FlatBufferBuilder
from edgeunicompile.mlir import MLIRInstaller, MLIRContext

__all__ = [
    "Context",
    "Graph",
    "Node",
    "Tensor",
    "PassManager",
    "PassBase",
    "ONNXConverter",
    "FlatBufferBuilder",
    "MLIRInstaller",
    "MLIRContext",
]
