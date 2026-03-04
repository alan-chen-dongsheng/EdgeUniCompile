# Python 和 C++ Pass 混合执行指南

本文档详细解释了 EdgeUniCompile 中 Python 和 C++ Pass 如何共存、互相调用，以及如何新增自定义 Pass。

## 目录

1. [架构概述](#架构概述)
2. [调用机制详解](#调用机制详解)
3. [如何新增 C++ Pass](#如何新增-c-pass)
4. [如何新增 Python Pass](#如何新增-python-pass)
5. [使用示例](#使用示例)
6. [常见问题](#常见问题)

---

## 架构概述

EdgeUniCompile 采用混合架构，允许 Python 和 C++ Pass 在同一编译流程中协同工作：

```
┌─────────────────────────────────────────────────────────────┐
│                      Python 层                               │
│                                                             │
│  ┌─────────────────┐         ┌─────────────────┐           │
│  │  Python Passes  │         │ HybridPassManager│           │
│  │  (纯 Python)     │         │ (统一执行器)     │           │
│  └─────────────────┘         └────────┬────────┘           │
│                                       │                     │
│                                       ▼                     │
│                              ┌─────────────────┐           │
│                              │  pybind11 模块   │           │
│                              │  (edgeunic_cpp) │           │
│                              └────────┬────────┘           │
└───────────────────────────────────────┼─────────────────────┘
                                        │ pybind11 绑定
                                        ▼
┌─────────────────────────────────────────────────────────────┐
│                       C++ 层                                 │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              C++ Passes (libedgeunic.so)             │   │
│  │  - PrintNodeNamesPass                                │   │
│  │  - MemoryAllocationPass                              │   │
│  │  - ConstantFoldingPass                               │   │
│  │  - InstructionScheduler                              │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### 核心组件

| 组件 | 位置 | 说明 |
|------|------|------|
| `HybridPassManager` | `python/edgeunicompile/passes/hybrid_executor.py` | 统一执行器，管理 Python 和 C++ Pass |
| `edgeunic_cpp` | `src/ffi/pybind_edgeunic.cpp` | pybind11 绑定的 C++ 模块 |
| `PassBase` (C++) | `include/edgeunicompile/passes/pass_base.h` | C++ Pass 基类 |
| `PassBase` (Python) | `python/edgeunicompile/passes/__init__.py` | Python Pass 基类 |

---

## 调用机制详解

### Python → C++ 调用流程

```python
# 1. Python 代码调用
manager.add_cpp_pass("print_node_names")
manager.run(graph)

# 2. HybridPassManager 内部流程
for pass_name, kwargs in self._cpp_passes:
    # 通过 pybind11 创建 C++ Pass 实例
    cpp_pass = edgeunic_cpp.PrintNodeNamesPass(**kwargs)

    # 执行 C++ Pass
    status = cpp_pass.run(cpp_graph, context)
```

### pybind11 绑定示例

```cpp
// src/ffi/pybind_edgeunic.cpp
#include <pybind11/pybind11.h>
#include "edgeunicompile/passes/print_node_names_pass.h"

namespace py = pybind11;

PYBIND11_MODULE(edgeunic_cpp, m) {
    m.doc() = "EdgeUniCompile C++ Passes";

    // 绑定 PrintNodeNamesPass
    py::class_<edgeunic::PrintNodeNamesPass,
               edgeunic::PassBase,
               std::shared_ptr<edgeunic::PrintNodeNamesPass>>(m, "PrintNodeNamesPass")
        .def(py::init<bool>(), py::arg("verbose") = false)
        .def("run", &edgeunic::PrintNodeNamesPass::Run,
             py::arg("graph"), py::arg("context") = nullptr);
}
```

### 数据传递

```
Python Graph 对象
       │
       │ pybind11 自动转换
       ▼
C++ GraphPtr (std::shared_ptr<Graph>)
       │
       │ Pass 执行
       ▼
C++ GraphPtr (修改后)
       │
       │ pybind11 自动转换
       ▼
Python Graph 对象 (已修改)
```

---

## 如何新增 C++ Pass

### 步骤 1: 创建头文件

在 `include/edgeunicompile/passes/` 创建新的头文件：

```cpp
// include/edgeunicompile/passes/my_custom_pass.h
#pragma once

#include "edgeunicompile/passes/pass_base.h"
#include "edgeunicompile/ir/graph.h"

namespace edgeunic {

/**
 * MyCustomPass - 示例自定义 Pass
 *
 * 描述这个 Pass 的功能
 */
class MyCustomPass : public PassBase {
public:
    MyCustomPass();
    ~MyCustomPass() override = default;

    Status Run(GraphPtr graph, std::shared_ptr<PassContext> context = nullptr) override;

    std::string GetDescription() const override;

private:
    // 私有辅助方法
    void ProcessNode(NodePtr node);
};

}  // namespace edgeunic
```

### 步骤 2: 创建实现文件

在 `src/passes/` 创建实现文件：

```cpp
// src/passes/my_custom_pass.cpp
#include "edgeunicompile/passes/my_custom_pass.h"
#include <iostream>

namespace edgeunic {

MyCustomPass::MyCustomPass() : PassBase("my_custom_pass") {}

std::string MyCustomPass::GetDescription() const {
    return "MyCustomPass: 描述你的 Pass 功能";
}

Status MyCustomPass::Run(GraphPtr graph, std::shared_ptr<PassContext> context) {
    if (!graph) {
        return Status::InvalidArgument("Graph cannot be null");
    }

    std::cout << "[MyCustomPass] Processing graph: " << graph->GetName() << std::endl;

    // 遍历所有节点并处理
    for (const auto& node : graph->GetNodes()) {
        ProcessNode(node);
    }

    // 更新上下文计数器
    if (context) {
        context->IncrementCounter("my_custom_pass_nodes_processed",
                                  static_cast<int64_t>(graph->GetNodes().size()));
    }

    return Status::Ok();
}

void MyCustomPass::ProcessNode(NodePtr node) {
    // 处理单个节点的逻辑
    std::cout << "  Processing node: " << node->GetName() << std::endl;
}

}  // namespace edgeunic
```

### 步骤 3: 更新 CMakeLists.txt

```cmake
# src/CMakeLists.txt
add_library(edgeunic SHARED
    # ... 其他文件
    passes/my_custom_pass.cpp  # 添加你的新文件
)
```

### 步骤 4: 添加 pybind11 绑定

```cpp
// src/ffi/pybind_edgeunic.cpp
#include "edgeunicompile/passes/my_custom_pass.h"

PYBIND11_MODULE(edgeunic_cpp, m) {
    // ... 其他绑定

    // 添加你的 Pass 绑定
    py::class_<edgeunic::MyCustomPass, edgeunic::PassBase,
               std::shared_ptr<edgeunic::MyCustomPass>>(m, "MyCustomPass")
        .def(py::init<>())
        .def("run", &edgeunic::MyCustomPass::Run,
             py::arg("graph"), py::arg("context") = nullptr);
}
```

### 步骤 5: 重建并测试

```bash
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# 测试
./tests/euc_test
```

---

## 如何新增 Python Pass

### 步骤 1: 创建 Pass 文件

在 `python/edgeunicompile/passes/` 创建新的 Pass：

```python
# python/edgeunicompile/passes/my_python_pass.py
#!/usr/bin/env python3.13
"""
MyPythonPass - 示例 Python Pass

描述这个 Pass 的功能
"""

from typing import Optional
from edgeunicompile.passes import PassBase
from edgeunicompile.core import Status, Context
from edgeunicompile.ir import Graph


class MyPythonPass(PassBase):
    """
    Python 实现的自定义 Pass

    优势:
    - 快速原型开发
    - 易于调试
    - 可以利用 Python 生态系统
    """

    def __init__(self, config_param: int = 10):
        """
        初始化 Pass

        Args:
            config_param: 配置参数示例
        """
        super().__init__("my_python_pass")
        self.config_param = config_param

    def run(self, graph: Graph, context: Context) -> Status:
        """
        执行 Pass

        Args:
            graph: 计算图
            context: 编译上下文

        Returns:
            Status 表示成功或失败
        """
        print(f"\n[MyPythonPass] Running on graph: {graph.name}")
        print(f"  Config param: {self.config_param}")

        # 遍历节点
        for node in graph.nodes:
            print(f"  Processing: {node.name} ({node.op_type})")

            # 在这里添加你的处理逻辑
            # ...

        # 更新上下文计数器
        context.increment_counter("my_python_pass_nodes", len(graph.nodes))

        return Status.ok()

    def check_prerequisites(self, graph: Graph) -> Status:
        """
        可选：检查执行前提条件

        Args:
            graph: 计算图

        Returns:
            Status 表示是否满足前提条件
        """
        if len(graph.nodes) == 0:
            return Status(error="Graph has no nodes")
        return Status.ok()
```

### 步骤 2: 更新 `__init__.py`

```python
# python/edgeunicompile/passes/__init__.py
from .my_python_pass import MyPythonPass

__all__ = [
    # ... 其他 Pass
    "MyPythonPass",
]
```

### 步骤 3: 使用 Pass

```python
from edgeunicompile.core import Context
from edgeunicompile.ir import Graph
from edgeunicompile.passes import MyPythonPass

context = Context()
pass_instance = MyPythonPass(config_param=20)

# 单独运行
status = pass_instance.run(graph, context)

# 或通过 HybridPassManager 运行
from edgeunicompile.passes.hybrid_executor import HybridPassManager

manager = HybridPassManager(context)
manager.add_pass(pass_instance)
result = manager.run(graph)
```

---

## 使用示例

### 示例 1: 基本使用

```python
from edgeunicompile.core import Context, Graph
from edgeunicompile.passes.hybrid_executor import HybridPassManager
from edgeunicompile.passes.tiling_pass import TilingPass

# 创建上下文和图
context = Context()
graph = Graph("my_model")
# ... 添加节点和 tensor ...

# 创建混合执行器
manager = HybridPassManager(context)

# 添加 Python Pass
manager.add_pass(TilingPass(tile_size=(64, 64)))

# 添加 C++ Pass
manager.add_cpp_pass("print_node_names", verbose=True)
manager.add_cpp_pass("memory_allocation", sram_max_size=2*1024*1024)

# 执行所有 Pass
result_graph = manager.run(graph)
```

### 示例 2: 交错执行

```python
# 定义执行顺序
execution_order = [
    ("python", "tiling_pass"),      # 先执行 Python Tiling
    ("cpp", "print_node_names"),    # 然后打印节点名
    ("cpp", "memory_allocation"),   # 然后分配内存
    ("python", "constant_folding"), # 最后常量折叠
]

result = manager.run_interleaved(graph, execution_order)
```

### 示例 3: 直接使用 C++ Pass

```python
from edgeunicompile.passes.hybrid_executor import (
    run_print_node_names,
    run_memory_allocation,
    generate_instructions
)

# 直接运行 C++ Pass
status = run_print_node_names(graph, verbose=True)

# 内存分配
status = run_memory_allocation(
    graph,
    sram_base=0,
    sram_max=3*1024*1024,  # 3MB
    dram_base=0,
    dram_max=5*1024*1024*1024  # 5GB
)

# 生成指令
scheduler = generate_instructions(graph)
scheduler.print_schedule()
```

### 示例 4: 完整的编译流程

```python
from edgeunicompile.core import Context
from edgeunicompile.ir import Graph
from edgeunicompile.onnx import ONNXConverter
from edgeunicompile.passes.hybrid_executor import HybridPassManager
from edgeunicompile.passes.tiling_pass import TilingPass

# 1. 从 ONNX 转换
graph = ONNXConverter.convert("model.onnx")

# 2. 创建上下文和执行器
context = Context()
manager = HybridPassManager(context)

# 3. 添加所有需要的 Pass
manager.add_pass(TilingPass(sram_limit_bytes=2*1024*1024))
manager.add_cpp_pass("constant_folding")
manager.add_cpp_pass("print_node_names")
manager.add_cpp_pass("memory_allocation")

# 4. 执行编译流程
optimized_graph = manager.run(graph)

# 5. 生成指令
from edgeunicompile.passes.hybrid_executor import generate_instructions
scheduler = generate_instructions(optimized_graph)
print(f"Generated {len(scheduler.get_packets())} instruction packets")
```

---

## 常见问题

### Q1: pybind11 模块加载失败怎么办？

```python
# 检查是否成功加载
from edgeunicompile.passes.hybrid_executor import CPP_MODULE_AVAILABLE
print(f"C++ module available: {CPP_MODULE_AVAILABLE}")

# 如果没有加载，检查:
# 1. pybind11 是否已安装
pip install pybind11

# 2. 是否已正确编译
cd build && cmake .. && make edgeunic_cpp
```

### Q2: 如何调试 C++ Pass？

```bash
# 使用 GDB 调试
gdb --args python -c "from edgeunicompile import edgeunic_cpp; ..."

# 或在 Python 中启用详细输出
from edgeunicompile.passes.hybrid_executor import CppPassExecutor
executor = CppPassExecutor()
# 查看可用的 Pass
print(executor.list_available_passes())
```

### Q3: Python 和 C++ Pass 的性能差异？

| 方面 | Python Pass | C++ Pass |
|------|-------------|----------|
| 执行速度 | 较慢 | 快 |
| 开发效率 | 高 | 中等 |
| 调试难度 | 低 | 中等 |
| 适用场景 | 原型、控制流 | 性能关键路径 |

### Q4: 如何在 Pass 之间传递数据？

所有 Pass 共享同一个 `Graph` 对象，修改会直接反映在图中：

```python
# Pass 1 修改图
def run(self, graph, context):
    graph.add_node(new_node)  # 添加节点
    return Status.ok()

# Pass 2 可以看到修改
def run(self, graph, context):
    # 能看到 Pass 1 添加的节点
    for node in graph.nodes:
        ...
```

### Q5: 如何添加带参数的 Pass？

```python
# Python Pass
class MyPass(PassBase):
    def __init__(self, param1: int, param2: str = "default"):
        super().__init__("my_pass")
        self.param1 = param1
        self.param2 = param2

# C++ Pass (通过 pybind11)
py::class_<MyPass>(m, "MyPass")
    .def(py::init<int, std::string>(),
         py::arg("param1"), py::arg("param2") = "default")

# 使用
manager.add_cpp_pass("my_pass", param1=42, param2="custom")
```

---

## 总结

EdgeUniCompile 的混合 Pass 系统提供了灵活的扩展机制：

- **Python Pass**: 适合快速原型、控制流复杂的场景
- **C++ Pass**: 适合性能关键、计算密集的场景
- **HybridPassManager**: 统一管理，支持交错执行

通过 pybind11，Python 和 C++ 之间的调用变得类型安全且高效。
