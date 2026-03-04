#!/usr/bin/env python3.13
"""
Tiling pass for EdgeUniCompile.

This module provides passes to tile operations to fit within
SRAM size limits. The tiling pass:
1. Slices input feature maps (IFM) into smaller regions
2. Runs multiple smaller convolutions on each slice
3. Concatenates the results (writing to DRAM to release SRAM)

This ensures that intermediate data fits within SRAM constraints.
"""

import math
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from edgeunicompile.passes import PassBase
from edgeunicompile.core import Status, Status as StatusCode, Shape, DataType
from edgeunicompile.ir import Graph, Node, Tensor


@dataclass
class TileConfig:
    """Configuration for a single tile."""
    tile_id: int
    tile_x: int
    tile_y: int

    # Output slice coordinates [y_start, y_end, x_start, x_end]
    out_y_start: int
    out_y_end: int
    out_x_start: int
    out_x_end: int

    # Input slice coordinates (accounting for kernel, stride, padding)
    in_y_start: int
    in_y_end: int
    in_x_start: int
    in_x_end: int

    # Actual tile dimensions
    tile_height: int
    tile_width: int


class TilingPass(PassBase):
    """
    Tile convolution operations to fit within SRAM size limit.

    This pass implements a slice-compute-concat pattern:
    1. Slice: Extract a region of the input feature map
    2. Compute: Run convolution on the sliced input
    3. Concat: Combine all tile outputs (writes to DRAM, releasing SRAM)

    Memory Flow:
    - Input IFM is in DRAM
    - For each tile:
      a. Slice input region -> SRAM buffer
      b. Conv compute -> SRAM buffer (output tile)
      c. (SRAM released after each tile)
    - Concat all output tiles -> DRAM (final output)
    """

    def __init__(self, tile_size=(64, 64), sram_limit_bytes: Optional[int] = None):
        """
        Initialize the tiling pass.

        Args:
            tile_size: Default tile size (width, height) in pixels.
            sram_limit_bytes: Optional SRAM limit in bytes. If None, uses context value.
        """
        super().__init__("tiling_pass")
        self.tile_size = tile_size
        self.sram_limit_bytes = sram_limit_bytes

    def check_needs_tiling(self, graph: Graph, context) -> Tuple[int, int]:
        """
        Check if any Conv2D nodes in the graph need tiling.

        Args:
            graph: The computation graph to check.
            context: The compilation context.

        Returns:
            Tuple of (nodes_needing_tiling, total_tiles_needed).
        """
        sram_size = self.sram_limit_bytes if self.sram_limit_bytes else getattr(context, 'sram_size', 2 * 1024 * 1024)

        nodes_needing_tiling = 0
        estimated_tiles = 0

        for node in graph.nodes:
            if node.op_type == "Conv2D":
                if not node.outputs:
                    continue
                output_tensor = node.outputs[0]

                # Calculate memory needed for output (float32 = 4 bytes)
                total_size = output_tensor.shape.num_elements() * 4

                # Also account for weights if they would exceed SRAM
                if len(node.inputs) > 1:
                    weights_size = node.inputs[1].shape.num_elements() * 4
                    total_size += weights_size

                if total_size > sram_size:
                    nodes_needing_tiling += 1

                    # Estimate number of tiles needed
                    out_height = output_tensor.shape.dims[2]
                    out_width = output_tensor.shape.dims[3]
                    tile_height, tile_width = self.tile_size
                    tiles_y = math.ceil(out_height / tile_height)
                    tiles_x = math.ceil(out_width / tile_width)
                    estimated_tiles += tiles_x * tiles_y

        return nodes_needing_tiling, estimated_tiles

    def run(self, graph: Graph, context):
        """
        Run the tiling pass on the graph.

        Args:
            graph: The computation graph to process.
            context: The compilation context.

        Returns:
            Status indicating success or failure.
        """
        print(f"\n{'='*60}")
        print(f"Running {self.name} with tile size: {self.tile_size}")

        sram_size = self.sram_limit_bytes if self.sram_limit_bytes else getattr(context, 'sram_size', 2 * 1024 * 1024)
        print(f"SRAM limit: {sram_size / (1024 * 1024):.1f}MB")

        # First check if any nodes need tiling
        nodes_needing_tiling, estimated_tiles = self.check_needs_tiling(graph, context)

        if nodes_needing_tiling == 0:
            print("No Conv2D nodes need tiling (all fit in SRAM)")
            return Status.ok()

        print(f"Found {nodes_needing_tiling} Conv2D nodes that need tiling")
        print(f"Estimated total tiles: {estimated_tiles}")

        # Collect Conv2D nodes that need tiling
        nodes_to_tile = []
        for node in graph.nodes:
            if node.op_type == "Conv2D":
                if not node.outputs:
                    continue
                output_tensor = node.outputs[0]
                # Calculate memory needed for output (float32 = 4 bytes)
                total_size = output_tensor.shape.num_elements() * 4

                # Also account for weights if they would exceed SRAM
                if len(node.inputs) > 1:
                    weights_size = node.inputs[1].shape.num_elements() * 4
                    total_size += weights_size

                if total_size > sram_size:
                    nodes_to_tile.append((node, total_size))

        # Process each node that needs tiling
        for node, total_size in nodes_to_tile:
            print(f"\n  Tiling node '{node.name}':")
            print(f"    Output size: {total_size / (1024*1024):.1f}MB")
            status = self._tile_conv2d(node, graph, context)
            if not status.is_ok():
                print(f"    ERROR: Failed to tile node: {status.message}")
                return status

        return Status.ok()

    def _calculate_tile_config(
        self,
        out_height: int,
        out_width: int,
        kernel_shape: List[int],
        strides: List[int],
        pads: List[int],
        in_height: int,
        in_width: int,
        tile_width: int,
        tile_height: int
    ) -> List[TileConfig]:
        """
        Calculate tile configurations for a convolution.

        Args:
            out_height: Output height
            out_width: Output width
            kernel_shape: Kernel size [kh, kw]
            strides: Strides [sy, sx]
            pads: Padding [pt, pb, pl, pr]
            in_height: Input height
            in_width: Input width
            tile_width: Target tile width
            tile_height: Target tile height

        Returns:
            List of TileConfig objects for each tile.
        """
        num_tiles_x = math.ceil(out_width / tile_width)
        num_tiles_y = math.ceil(out_height / tile_height)

        tile_configs = []
        tile_id = 0

        for tile_y in range(num_tiles_y):
            for tile_x in range(num_tiles_x):
                # Calculate output slice for this tile
                out_x_start = tile_x * tile_width
                out_x_end = min((tile_x + 1) * tile_width, out_width)
                out_y_start = tile_y * tile_height
                out_y_end = min((tile_y + 1) * tile_height, out_height)

                actual_tile_width = out_x_end - out_x_start
                actual_tile_height = out_y_end - out_y_start

                # Calculate input slice needed for this output region
                # Input region must include:
                # 1. The receptive field of the output region
                # 2. Padding on all sides

                pad_top, pad_bottom, pad_left, pad_right = pads[0], pads[1], pads[2], pads[3]
                kernel_h, kernel_w = kernel_shape[0], kernel_shape[1]
                stride_y, stride_x = strides[1], strides[0]

                # Effective kernel offset (how much the kernel extends before the output pixel)
                kernel_offset_y = (kernel_h - 1) // 2
                kernel_offset_x = (kernel_w - 1) // 2

                # Calculate input coordinates
                # The first output pixel at out_y_start needs input starting at:
                #   out_y_start * stride - pad_top
                # But we also need kernel_offset more pixels before that
                in_y_start = max(0, out_y_start * stride_y - pad_top)
                in_y_end = min(in_height, out_y_end * stride_y + pad_bottom + kernel_offset_y)
                in_x_start = max(0, out_x_start * stride_x - pad_left)
                in_x_end = min(in_width, out_x_end * stride_x + pad_right + kernel_offset_x)

                config = TileConfig(
                    tile_id=tile_id,
                    tile_x=tile_x,
                    tile_y=tile_y,
                    out_y_start=out_y_start,
                    out_y_end=out_y_end,
                    out_x_start=out_x_start,
                    out_x_end=out_x_end,
                    in_y_start=in_y_start,
                    in_y_end=in_y_end,
                    in_x_start=in_x_start,
                    in_x_end=in_x_end,
                    tile_height=actual_tile_height,
                    tile_width=actual_tile_width
                )
                tile_configs.append(config)
                tile_id += 1

        return tile_configs

    def _tile_conv2d(self, node: Node, graph: Graph, context) -> Status:
        """
        Split a Conv2D node into Slice -> Conv2D (per tile) -> Concat.

        Memory Flow:
        1. Original input IFM is in DRAM
        2. For each tile:
           - Slice node reads from DRAM, writes sliced IFM to SRAM
           - Conv2D reads sliced IFM + weights from SRAM, writes output tile to SRAM
           - After Conv2D, the sliced IFM can be freed in SRAM
        3. Concat reads all output tiles (from SRAM or DRAM) and writes final output to DRAM
        4. After Concat, all intermediate tile outputs are freed from SRAM

        Args:
            node: The Conv2D node to tile.
            graph: The computation graph.
            context: The compilation context.

        Returns:
            Status indicating success or failure.
        """
        # Get node attributes
        kernel_shape = node.get_attribute("kernel_shape", [3, 3])
        strides = node.get_attribute("strides", [1, 1])
        pads = node.get_attribute("pads", [0, 0, 0, 0])  # Default to no padding
        dilations = node.get_attribute("dilations", [1, 1])
        groups = node.get_attribute("groups", 1)

        # Ensure pads has 4 elements [pt, pb, pl, pr]
        if len(pads) == 2:
            pads = [pads[0], pads[0], pads[1], pads[1]]  # [py, py, px, px] -> [pt, pb, pl, pr]
        elif len(pads) != 4:
            pads = [0, 0, 0, 0]

        # Get input and output tensors
        input_tensor = node.inputs[0]  # NCHW format
        weight_tensor = node.inputs[1]
        bias_tensor = node.inputs[2] if len(node.inputs) > 2 else None
        output_tensor = node.outputs[0]

        # Get shapes (NCHW format)
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

        # Calculate tile configurations
        tile_configs = self._calculate_tile_config(
            out_height, out_width,
            kernel_shape, strides, pads,
            in_height, in_width,
            tile_width, tile_height
        )

        total_tiles = len(tile_configs)
        print(f"    Tile size: {tile_width}x{tile_height}")
        print(f"    Number of tiles: {total_tiles}")

        # Calculate memory requirements per tile
        max_tile_memory = 0
        for config in tile_configs:
            # Memory for sliced input
            sliced_input_elems = batch_size * in_channels * (config.in_y_end - config.in_y_start) * (config.in_x_end - config.in_x_start)
            # Memory for output tile
            output_tile_elems = batch_size * out_channels * config.tile_height * config.tile_width
            # Weights are shared (loaded once or streamed)

            tile_memory = (sliced_input_elems + output_tile_elems) * 4  # float32
            max_tile_memory = max(max_tile_memory, tile_memory)

        print(f"    Max memory per tile: ~{max_tile_memory / 1024:.1f}KB")

        # Store original node info for removal
        original_node = node
        original_output = output_tensor

        # Create intermediate tensors and nodes for each tile
        tile_output_tensors = []

        for config in tile_configs:
            # 1. Create Slice node to extract input region
            slice_input_tensor = Tensor(
                name=f"{input_tensor.name}_slice_{config.tile_id}",
                dtype=input_tensor.dtype,
                shape=Shape([
                    batch_size,
                    in_channels,
                    config.in_y_end - config.in_y_start,
                    config.in_x_end - config.in_x_start
                ])
            )
            # Note: memory location hints would be stored in graph metadata or context
            graph.add_tensor(slice_input_tensor)

            slice_node = Node(f"{node.name}_slice_{config.tile_id}", "Slice")
            slice_node.add_input(input_tensor)  # Original full input
            slice_node.add_output(slice_input_tensor)
            slice_node.set_attribute("starts", [0, 0, config.in_y_start, config.in_x_start])
            slice_node.set_attribute("ends", [batch_size, in_channels, config.in_y_end, config.in_x_end])
            slice_node.set_attribute("axes", [0, 1, 2, 3])
            slice_node.set_attribute("is_tiling_slice", True)
            slice_node.set_attribute("tile_id", config.tile_id)
            graph.add_node(slice_node)

            # 2. Create output tensor for this tile's convolution
            tile_output = Tensor(
                name=f"{original_output.name}_tile_{config.tile_id}",
                dtype=DataType.FLOAT32,
                shape=Shape([batch_size, out_channels, config.tile_height, config.tile_width])
            )
            # Note: SRAM hint - this tensor stays in SRAM until concat
            tile_output_tensors.append(tile_output)
            graph.add_tensor(tile_output)

            # 3. Create Conv2D node for this tile
            tile_node = Node(f"{node.name}_tile_{config.tile_id}", "Conv2D")
            tile_node.add_input(slice_input_tensor)  # SLICED input from SRAM
            tile_node.add_input(weight_tensor)  # Same weights (shared)
            if bias_tensor:
                tile_node.add_input(bias_tensor)
            tile_node.add_output(tile_output)

            # Conv2D attributes (adjusted for no additional padding since we sliced)
            # Note: We may need to adjust padding for edge tiles
            tile_pads = list(pads)  # Copy original pads

            # For edge tiles, we might need different padding
            # This is handled at runtime based on slice position
            tile_node.set_attribute("kernel_shape", kernel_shape)
            tile_node.set_attribute("strides", [1, 1])  # Stride is 1 since we already sliced
            tile_node.set_attribute("pads", [0, 0, 0, 0])  # No padding needed after slice
            tile_node.set_attribute("dilations", dilations)
            tile_node.set_attribute("groups", groups)

            # Tiling metadata
            tile_node.set_attribute("tile_id", config.tile_id)
            tile_node.set_attribute("tile_position", [config.tile_x, config.tile_y])
            tile_node.set_attribute("is_tiled", True)
            tile_node.set_attribute("output_slice", [
                config.out_y_start, config.out_y_end,
                config.out_x_start, config.out_x_end
            ])
            graph.add_node(tile_node)

            print(f"      Tile {config.tile_id}: "
                  f"slice[{config.in_y_start}:{config.in_y_end}, {config.in_x_start}:{config.in_x_end}] -> "
                  f"conv[{config.tile_height}x{config.tile_width}]")

        # 4. Create Concat node to combine all tile outputs
        # The Concat node writes to DRAM, which releases SRAM space
        concat_node = Node(f"{node.name}_concat", "Concat")
        concat_node.set_attribute("axis", 2)  # Concatenate along H dimension first
        concat_node.set_attribute("tile_count", total_tiles)
        concat_node.set_attribute("is_tiled_concat", True)
        concat_node.set_attribute("memory_location", "DRAM")  # Output goes to DRAM

        # Add tile outputs as inputs to concat (in row-major order for proper concatenation)
        # For 2D tiling, we need to concatenate along H first, then W
        # This requires careful ordering or multiple concat operations
        if total_tiles == 1:
            # Single tile - just copy
            concat_node.add_input(tile_output_tensors[0])
        else:
            # Multiple tiles - need 2D concatenation
            # Strategy: concat along H for each column, then concat columns along W

            num_tiles_x = math.ceil(out_width / tile_width)
            num_tiles_y = math.ceil(out_height / tile_height)

            if num_tiles_x == 1:
                # Only tiling in Y dimension - simple concat along axis 2 (H)
                for tile_tensor in tile_output_tensors:
                    concat_node.add_input(tile_tensor)
                concat_node.set_attribute("axis", 2)
            elif num_tiles_y == 1:
                # Only tiling in X dimension - concat along axis 3 (W)
                for tile_tensor in tile_output_tensors:
                    concat_node.add_input(tile_tensor)
                concat_node.set_attribute("axis", 3)
            else:
                # 2D tiling - need hierarchical concat
                # First concat along H for each X column, then concat results along W

                # Create intermediate concat tensors for each column
                column_concat_tensors = []
                for tx in range(num_tiles_x):
                    col_concat = Tensor(
                        name=f"{original_output.name}_col_concat_{tx}",
                        dtype=DataType.FLOAT32,
                        shape=Shape([batch_size, out_channels, out_height, tile_width if tx < num_tiles_x - 1 else (out_width - tx * tile_width)])
                    )
                    # Note: DRAM output - stored in graph metadata
                    graph.add_tensor(col_concat)
                    column_concat_tensors.append(col_concat)

                    # Create column concat node
                    col_concat_node = Node(f"{node.name}_col_concat_{tx}", "Concat")
                    col_concat_node.set_attribute("axis", 2)  # Concat along H

                    # Add tiles for this column
                    for ty in range(num_tiles_y):
                        tile_idx = ty * num_tiles_x + tx
                        if tile_idx < len(tile_output_tensors):
                            col_concat_node.add_input(tile_output_tensors[tile_idx])

                    col_concat_node.add_output(col_concat)
                    col_concat_node.set_attribute("memory_location", "DRAM")
                    graph.add_node(col_concat_node)

                # Final concat along W
                for col_tensor in column_concat_tensors:
                    concat_node.add_input(col_tensor)
                concat_node.set_attribute("axis", 3)  # Concat along W
                concat_node.set_attribute("is_hierarchical_concat", True)

        # Set concat output
        concat_node.add_output(original_output)
        graph.add_node(concat_node)

        # Remove the original Conv2D node
        graph.remove_node(original_node)

        # Update context counters
        context.increment_counter("conv2d_tiled", 1)
        context.increment_counter("conv2d_tiles_created", total_tiles)
        context.increment_counter("slice_nodes_created", total_tiles)
        context.increment_counter("concat_nodes_created", 1)

        print(f"    Created: {total_tiles} Slice + {total_tiles} Conv2D + 1 Concat nodes")
        print(f"    Memory flow: DRAM -> Slice(SRAM) -> Conv2D(SRAM) -> Concat -> DRAM")

        return Status.ok()


class SliceNodeCreator:
    """Helper class to create Slice nodes for tiling."""

    @staticmethod
    def create_slice_node(
        name: str,
        input_tensor: Tensor,
        output_tensor: Tensor,
        starts: List[int],
        ends: List[int],
        axes: Optional[List[int]] = None
    ) -> Node:
        """
        Create a Slice node.

        Args:
            name: Node name
            input_tensor: Input tensor to slice
            output_tensor: Output sliced tensor
            starts: Start indices for each dimension
            ends: End indices for each dimension
            axes: Axes to slice (default: [0, 1, 2, ...])

        Returns:
            Slice node
        """
        node = Node(name, "Slice")
        node.add_input(input_tensor)
        node.add_output(output_tensor)
        node.set_attribute("starts", starts)
        node.set_attribute("ends", ends)
        if axes:
            node.set_attribute("axes", axes)
        return node


class ConcatNodeCreator:
    """Helper class to create Concat nodes for tiling."""

    @staticmethod
    def create_concat_node(
        name: str,
        input_tensors: List[Tensor],
        output_tensor: Tensor,
        axis: int = 0
    ) -> Node:
        """
        Create a Concat node.

        Args:
            name: Node name
            input_tensors: Input tensors to concatenate
            output_tensor: Output concatenated tensor
            axis: Axis to concatenate along

        Returns:
            Concat node
        """
        node = Node(name, "Concat")
        node.set_attribute("axis", axis)
        for tensor in input_tensors:
            node.add_input(tensor)
        node.add_output(output_tensor)
        return node
