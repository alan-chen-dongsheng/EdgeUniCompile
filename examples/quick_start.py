#!/usr/bin/env python3.13
"""
EdgeUniCompile 快速开始指南

这个脚本演示了 EdgeUniCompile 的基本使用流程。
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import edgeunicompile as euc
from edgeunicompile.onnx import ONNXConverter
from edgeunicompile.flatbuf import FlatBufferBuilder
from edgeunicompile.passes import PassBase, PassManager
from edgeunicompile.core import Context, Status


def main():
    print("EdgeUniCompile 快速开始指南")
    print("=" * 60)

    # 1. 加载 ONNX 模型
    print("\n1. 加载 ONNX 模型...")
    import onnx
    onnx_model = onnx.load("single_conv_model.onnx")
    print(f"   模型名称：{onnx_model.graph.name}")

    # 2. 转换为 EdgeUniCompile Graph
    print("\n2. 转换为 EdgeUniCompile Graph...")
    context = Context()
    graph = ONNXConverter.convert(onnx_model, context)
    print(f"   Graph 名称：{graph.name}")
    print(f"   节点数：{len(graph.nodes)}")
    print(f"   Tensor 数：{len(graph.tensors)}")

    # 3. 保存到 FlatBuffer
    print("\n3. 保存到 FlatBuffer...")
    FlatBufferBuilder.save_to_file(graph, "quick_start.fb")
    print(f"   文件大小：{os.path.getsize('quick_start.fb')} 字节")

    # 4. 从 FlatBuffer 加载
    print("\n4. 从 FlatBuffer 加载...")
    loaded_graph = FlatBufferBuilder.load_from_file("quick_start.fb")
    print(f"   Graph 名称：{loaded_graph.name}")
    print(f"   节点数：{len(loaded_graph.nodes)}")

    # 5. 创建自定义 Pass
    print("\n5. 创建自定义 Pass...")

    class PrintNodePass(PassBase):
        """打印节点信息的 Pass."""

        def __init__(self):
            super().__init__("print_node_pass")

        def run(self, graph, context):
            print("   节点列表:")
            for node in graph.nodes:
                print(f"     - {node.name}: {node.op_type}")
            return Status.ok()

    # 6. 运行 Pass Manager
    print("\n6. 运行 Pass Manager...")
    pm = PassManager(context)
    pm.add_pass(PrintNodePass())

    from edgeunicompile.passes.tiling_pass import TilingPass
    pm.add_pass(TilingPass(tile_size=(64, 64)))

    optimized_graph = pm.run(loaded_graph)
    print(f"   优化后节点数：{len(optimized_graph.nodes)}")

    # 7. 导出优化后的 Graph
    print("\n7. 导出优化后的 Graph...")
    FlatBufferBuilder.save_to_file(optimized_graph, "quick_start_optimized.fb")
    print(f"   文件大小：{os.path.getsize('quick_start_optimized.fb')} 字节")

    # 8. 转换为 MLIR
    print("\n8. 转换为 MLIR...")
    from edgeunicompile.mlir import MLIRContext
    mlir_context = MLIRContext(context)
    mlir_module = mlir_context.compile(optimized_graph)
    print(f"   MLIR 模块创建成功!")

    # 9. 导出 MLIR
    print("\n9. 导出 MLIR...")
    with open("quick_start.mlir", "w") as f:
        f.write(mlir_module.mlir_str)
    print(f"   文件大小：{os.path.getsize('quick_start.mlir')} 字节")

    print("\n" + "=" * 60)
    print("快速开始指南完成!")
    print("\n生成的文件:")
    for path in ["quick_start.fb", "quick_start_optimized.fb", "quick_start.mlir"]:
        if os.path.exists(path):
            print(f"   {path}: {os.path.getsize(path)} 字节")

    return 0


if __name__ == "__main__":
    sys.exit(main())
