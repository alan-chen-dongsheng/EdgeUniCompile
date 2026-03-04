#!/usr/bin/env python3.13
"""
Tiling pass for EdgeUniCompile.

This module provides passes to tile operations to fit within
SRAM size limits. The tiling pass splits large convolution operations
into smaller tiles that can fit in SRAM.
"""

import math
from typing import List, Tuple, Dict
from edgeunicompile.passes import PassBase
from edgeunicompile.core import Status, Shape, DataType
from edgeunicompile.ir import Graph, Node, Tensor


class TilingPass(PassBase):
    """
    Tile convolution operations to fit within SRAM size limit.

    This pass splits large Conv2D operations into multiple smaller
    Conv2D operations that process tiles of the output. The tiles
    are then concatenated to produce the final output.
    """

    def __init__(self, tile_size=(64, 64), sram_limit_bytes=None):
        """
        Initialize the tiling pass.

        Args:
            tile_size: Default tile size (width, height) in pixels.
            sram_limit_bytes: Optional SRAM limit in bytes. If None, uses context value.
        """
        super().__init__("tiling_pass")
        self.tile_size = tile_size
        self.sram_limit_bytes = sram_limit_bytes

    def run(self, graph, context):
        """
        Run the tiling pass on the graph.

        Args:
            graph: The computation graph to process.
            context: The compilation context.

        Returns:
            Status indicating success or failure.
        """
        print(f"\nRunning {self.name} with tile size: {self.tile_size}")

        sram_size = self.sram_limit_bytes if self.sram_limit_bytes else context.sram_size
        print(f"SRAM limit: {sram_size / (1024 * 1024):.1f}MB")

        # Collect nodes that need tiling
        nodes_to_tile = []
        for node in graph.nodes:
            if node.op_type == "Conv2D":
                output_tensor = node.outputs[0]
                total_size = output_tensor.shape.num_elements() * 4  # float32
                if total_size > sram_size:
                    nodes_to_tile.append((node, total_size))

        if not nodes_to_tile:
            print("No Conv2D nodes need tiling (all fit in SRAM)")
            return Status.ok()

        print(f"Found {len(nodes_to_tile)} Conv2D nodes that need tiling")

        # Process each node that needs tiling
        for node, total_size in nodes_to_tile:
            print(f"\n  Tiling node '{node.name}':")
            print(f"    Output size: {total_size / (1024*1024):.1f}MB")
            self._tile_conv2d(node, graph, context)

        return Status.ok()

    def _tile_conv2d(self, node: Node, graph: Graph, context):
        """
        Split a Conv2D node into multiple tiled Conv2D nodes.

        Args:
            node: The Conv2D node to tile.
            graph: The computation graph.
            context: The compilation context.
        """
        # Get node attributes
        kernel_shape = node.get_attribute("kernel_shape", [3, 3])
        strides = node.get_attribute("strides", [1, 1])
        pads = node.get_attribute("pads", [1, 1, 1, 1])
        dilations = node.get_attribute("dilations", [1, 1])

        # Get input and output tensors
        input_tensor = node.inputs[0]  # NCHW format
        weight_tensor = node.inputs[1]
        bias_tensor = node.inputs[2] if len(node.inputs) > 2 else None
        output_tensor = node.outputs[0]

        # Get shapes
        batch_size = input_tensor.shape.dims[0]
        in_channels = input_tensor.shape.dims[1]
        in_height = input_tensor.shape.dims[2]
        in_width = input_tensor.shape.dims[3]

        out_channels = output_tensor.shape.dims[1]
        out_height = output_tensor.shape.dims[2]
        out_width = output_tensor.shape.dims[3]

        print(f"    Input shape: [{batch_size}, {in_channels}, {in_height}, {in_width}]")
        print(f"    Output shape: [{batch_size}, {out_channels}, {out_height}, {out_width}]")
        print(f"    Kernel: {kernel_shape}, Strides: {strides}, Pads: {pads}")

        # Calculate tile dimensions
        tile_width, tile_height = self.tile_size

        # Calculate number of tiles needed
        num_tiles_x = math.ceil(out_width / tile_width)
        num_tiles_y = math.ceil(out_height / tile_height)
        total_tiles = num_tiles_x * num_tiles_y

        print(f"    Tile size: {tile_width}x{tile_height}")
        print(f"    Number of tiles: {total_tiles} ({num_tiles_x} x {num_tiles_y})")

        # Calculate SRAM requirement per tile
        # Need: input tile + weights + bias + output tile
        tile_input_size = batch_size * in_channels * (tile_height + 2 * pads[0]) * (tile_width + 2 * pads[2]) * 4
        tile_output_size = batch_size * out_channels * tile_height * tile_width * 4
        weights_size = weight_tensor.shape.num_elements() * 4
        bias_size = bias_tensor.shape.num_elements() * 4 if bias_tensor else 0
        total_tile_memory = tile_input_size + tile_output_size + weights_size + bias_size

        print(f"    Memory per tile: ~{total_tile_memory / 1024:.1f}KB")

        # Create new tensors and nodes for each tile
        new_nodes = []
        new_tensors = []

        # Mark the original output tensor as needing concatenation
        # We'll create intermediate output tensors for each tile
        original_output_name = output_tensor.name

        # Create slice information for each tile
        tile_info = []
        for tile_y in range(num_tiles_y):
            for tile_x in range(num_tiles_x):
                # Calculate output slice for this tile
                out_start_x = tile_x * tile_width
                out_end_x = min((tile_x + 1) * tile_width, out_width)
                out_start_y = tile_y * tile_height
                out_end_y = min((tile_y + 1) * tile_height, out_height)

                actual_tile_width = out_end_x - out_start_x
                actual_tile_height = out_end_y - out_start_y

                # Calculate input slice (accounting for padding and kernel)
                kernel_offset = (kernel_shape[0] - 1) // 2
                pad_top, pad_bottom, pad_left, pad_right = pads[0], pads[1], pads[2], pads[3]

                # Input start/end accounting for stride and padding
                in_start_x = max(0, out_start_x * strides[0] - pad_left)
                in_end_x = min(in_width, out_end_x * strides[0] + pad_right + (kernel_shape[0] - 1))
                in_start_y = max(0, out_start_y * strides[1] - pad_top)
                in_end_y = min(in_height, out_end_y * strides[1] + pad_bottom + (kernel_shape[1] - 1))

                tile_info.append({
                    'tile_x': tile_x,
                    'tile_y': tile_y,
                    'out_start_x': out_start_x,
                    'out_end_x': out_end_x,
                    'out_start_y': out_start_y,
                    'out_end_y': out_end_y,
                    'in_start_x': in_start_x,
                    'in_end_x': in_end_x,
                    'in_start_y': in_start_y,
                    'in_end_y': in_end_y,
                    'tile_width': actual_tile_width,
                    'tile_height': actual_tile_height,
                })

        # Create new nodes and tensors for each tile
        for i, info in enumerate(tile_info):
            tile_name = f"{node.name}_tile_{i}"

            # Create sliced input tensor (conceptual - we use the same input with slice attributes)
            # Create output tensor for this tile
            tile_output = Tensor(
                name=f"{original_output_name}_tile_{i}",
                dtype=DataType.FLOAT32,
                shape=Shape([batch_size, out_channels, info['tile_height'], info['tile_width']])
            )
            new_tensors.append(tile_output)
            graph.add_tensor(tile_output)

            # Create tiled Conv2D node
            tile_node = Node(tile_name, "Conv2D")
            tile_node.add_input(input_tensor)  # Same input, will be sliced at runtime
            tile_node.add_input(weight_tensor)  # Same weights
            if bias_tensor:
                tile_node.add_input(bias_tensor)
            tile_node.add_output(tile_output)

            # Copy and modify attributes for the tile
            tile_node.set_attribute("kernel_shape", kernel_shape)
            tile_node.set_attribute("strides", strides)
            tile_node.set_attribute("pads", pads)
            tile_node.set_attribute("dilations", dilations)

            # Add tiling-specific attributes
            tile_node.set_attribute("tile_index", i)
            tile_node.set_attribute("tile_position", [info['tile_x'], info['tile_y']])
            tile_node.set_attribute("tile_output_slice", [
                info['out_start_y'], info['out_end_y'],
                info['out_start_x'], info['out_end_x']
            ])
            tile_node.set_attribute("tile_input_slice", [
                info['in_start_y'], info['in_end_y'],
                info['in_start_x'], info['in_end_x']
            ])
            tile_node.set_attribute("is_tiled", True)
            tile_node.set_attribute("concat_axis", [2, 3])  # H, W axes for concatenation

            new_nodes.append(tile_node)
            graph.add_node(tile_node)

            print(f"      Tile {i}: output[{info['out_start_y']}:{info['out_end_y']}, "
                  f"{info['out_start_x']}:{info['out_end_x']}] -> "
                  f"tile[{info['tile_height']}x{info['tile_width']}]")

        # Mark original node for removal (or keep it as a concat node)
        # We'll add a Concat node to combine all tile outputs
        concat_node = Node(f"{node.name}_concat", "Concat")
        concat_node.set_attribute("axis", 2)  # Concatenate along H axis first, then W

        # Add all tile outputs as inputs to concat
        for tile_node in new_nodes:
            concat_node.add_input(tile_node.outputs[0])

        # Note: output_tensor is already in the graph, don't add it again
        # Just set it as concat output
        concat_node.add_output(output_tensor)
        concat_node.set_attribute("tile_count", total_tiles)
        concat_node.set_attribute("tiles_x", num_tiles_x)
        concat_node.set_attribute("tiles_y", num_tiles_y)
        concat_node.set_attribute("is_tiled_concat", True)

        graph.add_node(concat_node)
        # output_tensor is already in the graph

        # Remove the original node
        graph.remove_node(node)

        # Update context counters
        context.increment_counter("conv2d_tiled", 1)
        context.increment_counter("conv2d_tiles_created", total_tiles)

        print(f"    Created {total_tiles} tiled Conv2D nodes + 1 Concat node")

    def _create_concat_pass(self, tile_nodes: List[Node], original_output: Tensor,
                            graph: Graph, concat_axis: int = 2) -> Node:
        """
        Create a Concat node to combine tiled outputs.

        Args:
            tile_nodes: List of tiled Conv2D nodes.
            original_output: Original output tensor.
            graph: The computation graph.
            concat_axis: Axis to concatenate along.

        Returns:
            The Concat node.
        """
        concat_node = Node(f"concat_{original_output.name}", "Concat")
        concat_node.set_attribute("axis", concat_axis)

        for tile_node in tile_nodes:
            concat_node.add_input(tile_node.outputs[0])

        concat_node.add_output(original_output)
        graph.add_node(concat_node)

        return concat_node
