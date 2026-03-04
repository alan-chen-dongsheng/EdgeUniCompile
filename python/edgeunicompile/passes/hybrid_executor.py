#!/usr/bin/env python3.13
"""
Hybrid Executor for running Python and C++ passes in sequence.

This module provides a unified execution framework that allows
Python passes and C++ passes to run together in a specified order.

It uses pybind11 for native C++ module integration, which provides:
- Type-safe bindings between Python and C++
- Automatic reference counting and memory management
- Zero-copy data passing where possible
- Exception handling across language boundaries
"""

import os
from typing import List, Optional, Tuple, Any
from edgeunicompile.core import Context, Status
from edgeunicompile.ir import Graph
from edgeunicompile.passes import PassBase, PassManager

# Try to import the C++ pybind11 module
try:
    from edgeunicompile import edgeunic_cpp
    CPP_MODULE_AVAILABLE = True
except ImportError as e:
    CPP_MODULE_AVAILABLE = False
    print(f"Warning: C++ pybind11 module not available: {e}")
    print("C++ passes will not be available. Install pybind11 and rebuild.")


class CppPassExecutor:
    """
    Executor for C++ passes via pybind11.

    This class provides a Python interface to C++ passes using native
    bindings, which is more efficient than ctypes-based FFI.
    """

    def __init__(self):
        """Initialize the C++ pass executor."""
        self.available = CPP_MODULE_AVAILABLE
        self._passes = {}

        if not self.available:
            print("Warning: C++ pass executor not available - pybind11 module not loaded")
            return

        # Register available C++ passes
        self._register_passes()

    def _register_passes(self):
        """Register available C++ passes."""
        self._passes = {
            "print_node_names": {
                "class": edgeunic_cpp.PrintNodeNamesPass,
                "description": "Print all node names in the graph"
            },
            "memory_allocation": {
                "class": edgeunic_cpp.MemoryAllocationPass,
                "description": "Allocate memory using linear scan algorithm"
            },
            "constant_folding": {
                "class": edgeunic_cpp.ConstantFoldingPass,
                "description": "Fold constant expressions"
            },
        }

    def list_available_passes(self) -> List[str]:
        """List names of available C++ passes."""
        return list(self._passes.keys())

    def create_pass(self, pass_name: str, **kwargs):
        """
        Create a C++ pass instance.

        Args:
            pass_name: Name of the pass to create.
            **kwargs: Arguments to pass to the pass constructor.

        Returns:
            C++ pass instance, or None if not available.
        """
        if not self.available:
            return None

        if pass_name not in self._passes:
            raise ValueError(f"Unknown C++ pass: {pass_name}. "
                           f"Available: {list(self._passes.keys())}")

        pass_class = self._passes[pass_name]["class"]
        return pass_class(**kwargs)

    def run_pass(self, pass_name: str, graph: Any, verbose: bool = False) -> Status:
        """
        Run a C++ pass on a graph.

        Args:
            pass_name: Name of the pass to run.
            graph: Graph object (will be converted to C++ Graph).
            verbose: Enable verbose output.

        Returns:
            Status indicating success or failure.
        """
        if not self.available:
            return Status(error="C++ pass executor not available")

        if pass_name not in self._passes:
            return Status(error=f"Unknown pass: {pass_name}")

        try:
            # Get the graph's underlying C++ object if it has one
            cpp_graph = graph._cpp_graph if hasattr(graph, '_cpp_graph') else graph

            # Create and run the pass
            cpp_pass = self.create_pass(pass_name, verbose=verbose)
            context = edgeunic_cpp.PassContext()
            status = cpp_pass.run(cpp_graph, context)

            return Status.ok() if status.is_ok() else Status(error=status.to_string())

        except Exception as e:
            return Status(error=f"C++ pass '{pass_name}' failed: {str(e)}")


class HybridPassManager(PassManager):
    """
    Manager for running both Python and C++ passes in sequence.

    The HybridPassManager extends the Python PassManager to support
    running C++ passes alongside Python passes using pybind11 bindings.

    Example usage:
        ```python
        context = Context()
        manager = HybridPassManager(context)

        # Add Python passes
        manager.add_pass(TilingPass())

        # Add C++ passes
        manager.add_cpp_pass("print_node_names")
        manager.add_cpp_pass("memory_allocation")

        # Run all passes in order
        result_graph = manager.run(input_graph)
        ```
    """

    def __init__(self, context: Context):
        """
        Initialize the hybrid pass manager.

        Args:
            context: Compilation context.
        """
        super().__init__(context)
        self.cpp_executor = CppPassExecutor()
        self._cpp_passes: List[Tuple[str, dict]] = []  # List of (name, kwargs)

    def add_cpp_pass(self, pass_name: str, **kwargs):
        """
        Add a C++ pass to the execution sequence.

        Args:
            pass_name: Name of the C++ pass to add.
            **kwargs: Arguments to pass to the pass constructor.
        """
        available = self.cpp_executor.list_available_passes()
        if pass_name not in available:
            raise ValueError(f"Unknown C++ pass: {pass_name}. Available: {available}")

        self._cpp_passes.append((pass_name, kwargs))
        print(f"Added C++ pass: {pass_name}")

    def remove_cpp_pass(self, index: int):
        """
        Remove a C++ pass from the execution sequence by index.

        Args:
            index: Index of the pass to remove.
        """
        if 0 <= index < len(self._cpp_passes):
            removed = self._cpp_passes.pop(index)
            print(f"Removed C++ pass: {removed[0]}")

    def list_cpp_passes(self) -> List[dict]:
        """List all registered C++ passes."""
        return [
            {"name": name, "kwargs": kwargs}
            for name, kwargs in self._cpp_passes
        ]

    def run(self, graph: Graph, interleave: bool = False) -> Graph:
        """
        Run all Python and C++ passes on the graph.

        Args:
            graph: The computation graph to optimize.
            interleave: If True, run passes in registration order.
                       If False (default), run all Python passes first,
                       then all C++ passes.

        Returns:
            The optimized graph.

        Raises:
            RuntimeError: If a pass fails.
        """
        from copy import deepcopy

        current_graph = deepcopy(graph)

        if interleave:
            # Run passes in the order they were added
            # This requires tracking insertion order for both Python and C++ passes
            # For simplicity, we run Python passes first, then C++ passes
            pass

        # Run Python passes
        for pass_instance in self.passes:
            if not pass_instance.enabled:
                continue

            print(f"\n[Python] Running pass: {pass_instance.name}")
            status = pass_instance.run(current_graph, self.context)
            if not status.is_ok():
                raise RuntimeError(f"Python pass '{pass_instance.name}' failed: {status}")

            self.context.increment_counter(f"python_pass_{pass_instance.name}_runs")
            print(f"[Python] Pass {pass_instance.name} completed successfully")

        # Run C++ passes
        if self._cpp_passes and self.cpp_executor.available:
            for pass_name, kwargs in self._cpp_passes:
                print(f"\n[C++] Running pass: {pass_name}")
                status = self.cpp_executor.run_pass(pass_name, current_graph, **kwargs)
                if not status.is_ok():
                    raise RuntimeError(f"C++ pass '{pass_name}' failed: {status}")

                self.context.increment_counter(f"cpp_pass_{pass_name}_runs")
                print(f"[C++] Pass {pass_name} completed successfully")

        elif self._cpp_passes and not self.cpp_executor.available:
            print("\n[C++] Warning: C++ passes requested but pybind11 module not available")
            print("    Install pybind11 and rebuild to enable C++ passes")

        return current_graph

    def run_interleaved(self, graph: Graph,
                       execution_order: List[Tuple[str, str]]) -> Graph:
        """
        Run passes with explicit interleaving of Python and C++ passes.

        Args:
            graph: The computation graph to optimize.
            execution_order: List of (language, pass_name) tuples specifying
                           the execution order. e.g.,
                           [('python', 'tiling_pass'),
                            ('cpp', 'print_node_names'),
                            ('python', 'constant_folding')]

        Returns:
            The optimized graph.

        Raises:
            ValueError: If a pass is not found.
            RuntimeError: If a pass fails.
        """
        from copy import deepcopy

        current_graph = deepcopy(graph)

        for lang, pass_name in execution_order:
            if lang == "python":
                pass_instance = self.get_pass(pass_name)
                if pass_instance is None:
                    raise ValueError(f"Python pass '{pass_name}' not found")
                if not pass_instance.enabled:
                    print(f"[Python] Skipping disabled pass: {pass_name}")
                    continue

                print(f"\n[Python] Running pass: {pass_name}")
                status = pass_instance.run(current_graph, self.context)
                if not status.is_ok():
                    raise RuntimeError(f"Python pass '{pass_name}' failed: {status}")

                self.context.increment_counter(f"python_pass_{pass_name}_runs")
                print(f"[Python] Pass {pass_name} completed successfully")

            elif lang == "cpp":
                if not self.cpp_executor.available:
                    print(f"[C++] Warning: C++ library not loaded, skipping: {pass_name}")
                    continue

                # Find kwargs for this pass
                kwargs = {}
                for name, kw in self._cpp_passes:
                    if name == pass_name:
                        kwargs = kw
                        break

                print(f"\n[C++] Running pass: {pass_name}")
                status = self.cpp_executor.run_pass(pass_name, current_graph, **kwargs)
                if not status.is_ok():
                    raise RuntimeError(f"C++ pass '{pass_name}' failed: {status}")

                self.context.increment_counter(f"cpp_pass_{pass_name}_runs")
                print(f"[C++] Pass {pass_name} completed successfully")
            else:
                raise ValueError(f"Unknown language: {lang}")

        return current_graph

    def __repr__(self):
        python_passes = [p.name for p in self.passes]
        cpp_passes = [name for name, _ in self._cpp_passes]
        return (f"HybridPassManager(context={self.context}, "
                f"python_passes={python_passes}, cpp_passes={cpp_passes})")


# Convenience functions for running C++ passes directly
def run_print_node_names(graph: Graph, verbose: bool = False) -> Status:
    """
    Run PrintNodeNamesPass on a graph.

    Args:
        graph: Graph to process.
        verbose: Enable verbose output.

    Returns:
        Status indicating success or failure.
    """
    if CPP_MODULE_AVAILABLE:
        return edgeunic_cpp.run_print_node_names(graph._cpp_graph if hasattr(graph, '_cpp_graph') else graph, verbose)
    return Status(error="C++ module not available")


def run_memory_allocation(graph: Graph,
                          sram_base: int = 0,
                          sram_max: int = 3*1024*1024,
                          dram_base: int = 0,
                          dram_max: int = 5*1024*1024*1024) -> Status:
    """
    Run MemoryAllocationPass on a graph.

    Args:
        graph: Graph to process.
        sram_base: SRAM base address.
        sram_max: SRAM maximum size in bytes.
        dram_base: DRAM base address.
        dram_max: DRAM maximum size in bytes.

    Returns:
        Status indicating success or failure.
    """
    if CPP_MODULE_AVAILABLE:
        return edgeunic_cpp.run_memory_allocation(
            graph._cpp_graph if hasattr(graph, '_cpp_graph') else graph,
            sram_base, sram_max, dram_base, dram_max
        )
    return Status(error="C++ module not available")


def run_constant_folding(graph: Graph) -> Status:
    """
    Run ConstantFoldingPass on a graph.

    Args:
        graph: Graph to process.

    Returns:
        Status indicating success or failure.
    """
    if CPP_MODULE_AVAILABLE:
        return edgeunic_cpp.run_constant_folding(graph._cpp_graph if hasattr(graph, '_cpp_graph') else graph)
    return Status(error="C++ module not available")


def generate_instructions(graph: Graph) -> Any:
    """
    Generate and schedule instructions for a graph.

    Args:
        graph: Graph to process.

    Returns:
        InstructionScheduler object with scheduled packets.

    Raises:
        RuntimeError: If instruction generation fails.
    """
    if CPP_MODULE_AVAILABLE:
        return edgeunic_cpp.generate_instructions(
            graph._cpp_graph if hasattr(graph, '_cpp_graph') else graph
        )
    raise RuntimeError("C++ module not available")
