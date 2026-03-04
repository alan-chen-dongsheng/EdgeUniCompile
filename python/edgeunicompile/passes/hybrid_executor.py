#!/usr/bin/env python3.13
"""
Hybrid Executor for running Python and C++ passes in sequence.

This module provides a unified execution framework that allows
Python passes and C++ passes to run together in a specified order.
"""

import ctypes
import os
from typing import List, Optional, Any
from edgeunicompile.core import Context, Status
from edgeunicompile.ir import Graph
from edgeunicompile.passes import PassBase, PassManager
from edgeunicompile.flatbuf import FlatBufferBuilder


class CppPassExecutor:
    """
    Executor for C++ passes via FFI.

    This class loads the C++ library and provides an interface
    to run C++ passes from Python.
    """

    def __init__(self, library_path: str = None):
        """
        Initialize the C++ pass executor.

        Args:
            library_path: Path to the C++ library (libedgeunic.so).
                         If None, searches common locations.
        """
        self.lib = None
        self._load_library(library_path)

    def _load_library(self, library_path: str = None):
        """Load the C++ library."""
        possible_paths = [
            library_path,
            "./build/lib/libedgeunic.so",
            "../build/lib/libedgeunic.so",
            "libedgeunic.so",
        ]

        for path in possible_paths:
            if path and os.path.exists(path):
                try:
                    self.lib = ctypes.CDLL(path)
                    print(f"Loaded C++ library from: {path}")
                    return
                except OSError as e:
                    print(f"Failed to load {path}: {e}")
                    continue

        print("Warning: C++ library not found. C++ passes will be skipped.")

    def create_pass(self, pass_name: str) -> Optional[Any]:
        """
        Create a C++ pass instance.

        Args:
            pass_name: Name of the pass to create.

        Returns:
            Opaque handle to the C++ pass, or None if failed.
        """
        if self.lib is None:
            return None

        try:
            # TODO: Implement FFI for creating C++ passes
            # This requires defining C ABI-compatible functions in C++
            return None
        except Exception as e:
            print(f"Failed to create C++ pass '{pass_name}': {e}")
            return None

    def run_pass(self, pass_name: str, graph_bytes: bytes) -> bytes:
        """
        Run a C++ pass on a serialized graph.

        Args:
            pass_name: Name of the pass to run.
            graph_bytes: Serialized graph (FlatBuffer format).

        Returns:
            Modified graph bytes after running the pass.
        """
        if self.lib is None:
            raise RuntimeError("C++ library not loaded")

        # TODO: Implement FFI for running C++ passes
        # This requires:
        # 1. C-compatible function in C++ that takes graph bytes
        # 2. Returns modified graph bytes
        # 3. Proper memory management

        raise NotImplementedError("C++ pass execution via FFI not yet implemented")


class HybridPassManager(PassManager):
    """
    Manager for running both Python and C++ passes in sequence.

    The HybridPassManager extends the Python PassManager to support
    running C++ passes alongside Python passes. It uses FlatBuffer
    serialization to pass data between Python and C++.
    """

    def __init__(self, context: Context, cpp_library_path: str = None):
        """
        Initialize the hybrid pass manager.

        Args:
            context: Compilation context.
            cpp_library_path: Optional path to C++ library.
        """
        super().__init__(context)
        self.cpp_executor = CppPassExecutor(cpp_library_path)
        self._cpp_passes = []  # List of C++ pass names

    def add_cpp_pass(self, pass_name: str):
        """
        Add a C++ pass to the execution sequence.

        Args:
            pass_name: Name of the C++ pass to add.
        """
        self._cpp_passes.append(pass_name)
        print(f"Added C++ pass: {pass_name}")

    def remove_cpp_pass(self, pass_name: str):
        """
        Remove a C++ pass from the execution sequence.

        Args:
            pass_name: Name of the C++ pass to remove.
        """
        if pass_name in self._cpp_passes:
            self._cpp_passes.remove(pass_name)
            print(f"Removed C++ pass: {pass_name}")

    def run(self, graph: Graph) -> Graph:
        """
        Run all enabled Python and C++ passes on the graph.

        Passes are executed in the order they were added, with
        Python and C++ passes interleaved as specified.

        Args:
            graph: The computation graph to optimize.

        Returns:
            The optimized graph.

        Raises:
            RuntimeError: If a pass fails.
        """
        from copy import deepcopy

        current_graph = deepcopy(graph)

        # Alternate between Python and C++ passes
        # For simplicity, we run all Python passes first, then C++ passes
        # TODO: Implement proper interleaving based on registration order

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

        # Serialize graph to FlatBuffer for C++ passes
        if self._cpp_passes and self.cpp_executor.lib is not None:
            graph_bytes = FlatBufferBuilder.build(current_graph, self.context)

            # Run C++ passes
            for pass_name in self._cpp_passes:
                print(f"\n[C++] Running pass: {pass_name}")
                try:
                    # TODO: Implement actual FFI call
                    # graph_bytes = self.cpp_executor.run_pass(pass_name, graph_bytes)
                    print(f"[C++] Pass {pass_name} completed (placeholder)")
                    self.context.increment_counter(f"cpp_pass_{pass_name}_runs")
                except Exception as e:
                    raise RuntimeError(f"C++ pass '{pass_name}' failed: {e}")

            # Deserialize graph back from FlatBuffer
            current_graph = FlatBufferBuilder.parse(graph_bytes, self.context)

        return current_graph

    def run_interleaved(self, graph: Graph, execution_order: List[tuple]) -> Graph:
        """
        Run passes with explicit interleaving of Python and C++ passes.

        Args:
            graph: The computation graph to optimize.
            execution_order: List of (language, pass_name) tuples specifying
                            the execution order. e.g., [('python', 'tiling_pass'),
                            ('cpp', 'print_nodes_pass'), ('python', 'other_pass')]

        Returns:
            The optimized graph.
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
                if self.cpp_executor.lib is None:
                    print(f"[C++] Warning: C++ library not loaded, skipping pass: {pass_name}")
                    continue

                print(f"\n[C++] Running pass: {pass_name}")

                # Serialize graph to FlatBuffer
                graph_bytes = FlatBufferBuilder.build(current_graph, self.context)

                # TODO: Implement actual FFI call
                # graph_bytes = self.cpp_executor.run_pass(pass_name, graph_bytes)
                print(f"[C++] Pass {pass_name} completed (placeholder)")

                # Deserialize graph back
                current_graph = FlatBufferBuilder.parse(graph_bytes, self.context)

                self.context.increment_counter(f"cpp_pass_{pass_name}_runs")
            else:
                raise ValueError(f"Unknown language: {lang}")

        return current_graph

    def list_passes(self) -> List[dict]:
        """List all registered passes (Python and C++)."""
        result = super().list_passes()
        for pass_name in self._cpp_passes:
            result.append({
                "name": pass_name,
                "enabled": True,
                "type": "C++"
            })
        return result

    def __repr__(self):
        python_passes = [p.name for p in self.passes]
        return f"HybridPassManager(context={self.context}, python_passes={python_passes}, cpp_passes={self._cpp_passes})"


def create_hybrid_pass_manager(context: Context,
                                cpp_library_path: str = None,
                                python_passes: List[PassBase] = None,
                                cpp_passes: List[str] = None) -> HybridPassManager:
    """
    Create a hybrid pass manager with both Python and C++ passes.

    Args:
        context: Compilation context.
        cpp_library_path: Optional path to C++ library.
        python_passes: List of Python pass instances to add.
        cpp_passes: List of C++ pass names to add.

    Returns:
        Configured HybridPassManager instance.
    """
    manager = HybridPassManager(context, cpp_library_path)

    if python_passes:
        for pass_instance in python_passes:
            manager.add_pass(pass_instance)

    if cpp_passes:
        for pass_name in cpp_passes:
            manager.add_cpp_pass(pass_name)

    return manager
