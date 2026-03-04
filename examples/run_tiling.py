#!/usr/bin/env python3.13

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import edgeunicompile as euc
from edgeunicompile.passes import PassManager
from edgeunicompile.passes.tiling_pass import TilingPass
from edgeunicompile.core import Context


def main():
    print("EdgeUniCompile - Tiling Pass")
    print("=" * 50)

    # Load graph
    input_path = "single_conv_model.fb"
    if not os.path.exists(input_path):
        print(f"File not found: {input_path}")
        sys.exit(1)
    graph = euc.FlatBufferBuilder.load_from_file(input_path)

    # Create context with SRAM limit of 3MB
    context = Context(sram_size=3 * 1024 * 1024)
    print(f"SRAM limit set to {context.sram_size / (1024 * 1024):.1f}MB")

    # Create pass manager
    pm = PassManager(context)
    pm.add_pass(TilingPass(tile_size=(64, 64)))

    print(f"Pass manager created with {len(pm.passes)} passes")

    # Run passes and print context counters
    optimized_graph = pm.run(graph)
    print()

    print(f"Context counters after passes: {context.get_all_counters()}")
    if hasattr(context, "tiling_tile_size"):
        print(f"Tiling tile size: {context.tiling_tile_size}")

    # Check if tiling was applied
    for node in optimized_graph.nodes:
        if node.has_attribute("tiling"):
            print(f"\nTiling applied to node {node.name}: {node.attributes['tiling']}")


if __name__ == "__main__":
    main()
