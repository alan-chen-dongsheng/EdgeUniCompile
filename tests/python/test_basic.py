"""
Test basic functionality of the package.
"""

import pytest
import edgeunicompile as euc


def test_import():
    """Test that the package imports correctly."""
    assert euc is not None


def test_version():
    """Test version attribute."""
    assert hasattr(euc, '__version__')
    assert isinstance(euc.__version__, str)
    assert len(euc.__version__) > 0


def test_context():
    """Test context creation and properties."""
    ctx = euc.Context()
    assert ctx is not None
    assert ctx.opt_level == 3
    assert ctx.sram_size == 32 * 1024 * 1024
    assert ctx.target_arch == 'armv8'


def test_graph_creation():
    """Test graph creation."""
    from edgeunicompile.ir import Graph

    graph = Graph('test_graph')
    assert graph.name == 'test_graph'
    assert len(graph.nodes) == 0
    assert len(graph.tensors) == 0


def test_node_creation():
    """Test node creation."""
    from edgeunicompile.ir import Node

    node = Node('test_node', 'Add')
    assert node.name == 'test_node'
    assert node.op_type == 'Add'
    assert len(node.inputs) == 0
    assert len(node.outputs) == 0


def test_tensor_creation():
    """Test tensor creation."""
    from edgeunicompile.ir import Tensor

    tensor = Tensor('test_tensor', 'float32', (2, 3))
    assert tensor.name == 'test_tensor'
    assert tensor.dtype == 'float32'
    assert tensor.shape == (2, 3)
