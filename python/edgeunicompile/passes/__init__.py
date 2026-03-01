"""
Pass infrastructure supporting both C++ and Python passes.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Callable, Any
from abc import ABC, abstractmethod
from edgeunicompile.core import Context, Status


class PassBase(ABC):
    """
    Base class for all passes in the EdgeUniCompile system.

    Passes can be implemented in either C++ or Python and are used to
    optimize, transform, or analyze computation graphs.
    """

    def __init__(self, name: str):
        """
        Create a new pass.

        Args:
            name: Name of the pass.
        """
        self._name = name
        self._enabled = True

    @property
    def name(self) -> str:
        """Get the pass name."""
        return self._name

    @property
    def enabled(self) -> bool:
        """Check if the pass is enabled."""
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool):
        """Set whether the pass is enabled."""
        self._enabled = value

    @abstractmethod
    def run(self, graph: "Graph", context: Context) -> Status:
        """
        Run the pass on the given graph.

        Args:
            graph: The computation graph to process.
            context: The compilation context.

        Returns:
            Status indicating success or failure.
        """
        pass


@dataclass
class PassManager:
    """
    Manager for running multiple passes in sequence.

    The PassManager orchestrates the execution of passes, providing:
        - Pass registration and ordering
        - Pass configuration
        - Execution pipeline
    """

    context: Context
    passes: List[PassBase] = field(default_factory=list)
    _pass_map: dict = field(default_factory=dict, init=False)

    def __post_init__(self):
        """Initialize the pass manager."""
        for pass_instance in self.passes:
            self._pass_map[pass_instance.name] = pass_instance

    def add_pass(self, pass_instance: PassBase):
        """
        Add a pass to the manager.

        Args:
            pass_instance: The pass to add.
        """
        self.passes.append(pass_instance)
        self._pass_map[pass_instance.name] = pass_instance

    def remove_pass(self, pass_name: str):
        """
        Remove a pass from the manager.

        Args:
            pass_name: Name of the pass to remove.
        """
        if pass_name in self._pass_map:
            for i, pass_instance in enumerate(self.passes):
                if pass_instance.name == pass_name:
                    del self.passes[i]
                    del self._pass_map[pass_name]
                    break

    def get_pass(self, pass_name: str) -> Optional[PassBase]:
        """
        Get a pass by name.

        Args:
            pass_name: Name of the pass to get.

        Returns:
            The pass instance or None if not found.
        """
        return self._pass_map.get(pass_name)

    def run(self, graph: "Graph") -> "Graph":
        """
        Run all enabled passes on the graph.

        Args:
            graph: The computation graph to optimize.

        Returns:
            The optimized graph.
        """
        from copy import deepcopy

        optimized_graph = deepcopy(graph)

        for pass_instance in self.passes:
            if not pass_instance.enabled:
                continue

            status = pass_instance.run(optimized_graph, self.context)
            if not status.is_ok():
                raise RuntimeError(f"Pass '{pass_instance.name}' failed: {status}")

            self.context.increment_counter(f"pass_{pass_instance.name}_runs")

        return optimized_graph

    def run_pass(self, pass_name: str, graph: "Graph") -> "Graph":
        """
        Run a specific pass.

        Args:
            pass_name: Name of the pass to run.
            graph: The computation graph to optimize.

        Returns:
            The optimized graph.

        Raises:
            ValueError: If pass is not found.
        """
        if pass_name not in self._pass_map:
            raise ValueError(f"Pass '{pass_name}' not found")

        pass_instance = self._pass_map[pass_name]

        if not pass_instance.enabled:
            raise ValueError(f"Pass '{pass_name}' is not enabled")

        optimized_graph = deepcopy(graph)
        status = pass_instance.run(optimized_graph, self.context)
        if not status.is_ok():
            raise RuntimeError(f"Pass '{pass_name}' failed: {status}")

        self.context.increment_counter(f"pass_{pass_instance.name}_runs")

        return optimized_graph

    def disable_pass(self, pass_name: str):
        """
        Disable a pass.

        Args:
            pass_name: Name of the pass to disable.

        Raises:
            ValueError: If pass is not found.
        """
        if pass_name not in self._pass_map:
            raise ValueError(f"Pass '{pass_name}' not found")
        self._pass_map[pass_name].enabled = False

    def enable_pass(self, pass_name: str):
        """
        Enable a pass.

        Args:
            pass_name: Name of the pass to enable.

        Raises:
            ValueError: If pass is not found.
        """
        if pass_name not in self._pass_map:
            raise ValueError(f"Pass '{pass_name}' not found")
        self._pass_map[pass_name].enabled = True

    def list_passes(self) -> List[dict]:
        """
        List all registered passes with their status.

        Returns:
            List of pass information dictionaries.
        """
        result = []
        for pass_instance in self.passes:
            result.append({
                "name": pass_instance.name,
                "enabled": pass_instance.enabled,
                "type": type(pass_instance).__name__
            })
        return result

    def __repr__(self):
        return f"PassManager(context={self.context}, passes={[p.name for p in self.passes]})"

    def __str__(self):
        return self.__repr__()


# Convenience function to create a default pass manager
def create_default_pass_manager(context: Context) -> PassManager:
    """
    Create a default pass manager with standard passes.

    Args:
        context: The compilation context.

    Returns:
        A PassManager instance with default passes.
    """
    manager = PassManager(context)

    # Add default passes
    from . import default_passes

    for pass_cls in default_passes.get_default_passes():
        manager.add_pass(pass_cls())

    return manager
