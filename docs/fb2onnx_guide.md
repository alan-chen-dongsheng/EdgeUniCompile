# FlatBuffer 到 ONNX 转换工具

这个工具将 EdgeUniCompile 的 FlatBuffer 格式转换为 ONNX 格式，以便使用 Netron 进行可视化。

## 为什么需要转换？

Netron 是一个流行的神经网络模型可视化工具，但它不直接支持 EdgeUniCompile 的自定义 FlatBuffer 格式。通过转换为 ONNX 格式，您可以：

1. 在 Netron 中可视化模型结构
2. 查看每一层的连接关系
3. 检查张量形状和属性
4. 调试和验证模型

## 使用方法

### 单个文件转换

```bash
# 基本用法
uv run python examples/fb2onnx.py <input.fb> [output.onnx]

# 示例
uv run python examples/fb2onnx.py tiling_output.fb
uv run python examples/fb2onnx.py tiling_output.fb my_model.onnx
```

如果不指定输出文件名，将自动生成 `<input>_for_netron.onnx`。

### 批量转换

```bash
# 转换目录下所有 .fb 文件
uv run python examples/fb2onnx.py --batch <input_dir> [output_dir]

# 示例
uv run python examples/fb2onnx.py --batch . converted_onnx
```

## 在 Netron 中查看

### 方法 1: 使用 Netron 网站
1. 打开 https://netron.app/
2. 点击 "Open Model" 或直接拖拽 `.onnx` 文件
3. 浏览模型结构

### 方法 2: 本地安装 Netron
```bash
# 使用 npm 安装
npm install -g netron

# 打开文件
netron tiling_output_for_netron.onnx
```

### 方法 3: 使用 Python
```bash
pip install netron
netron tiling_output_for_netron.onnx
```

## 转换示例

### 示例 1: 转换单卷积模型

```bash
# 1. 创建 ONNX 模型
uv run python examples/create_conv_model.py

# 2. 转换为 FlatBuffer
uv run python examples/convert_onnx_to_flatbuffer.py single_conv_model.onnx single_conv_model.fb

# 3. 运行 Tiling Pass
uv run python examples/run_tiling.py

# 4. 转换为 ONNX 用于可视化
uv run python examples/fb2onnx.py tiling_output.fb

# 5. 在 Netron 中打开
netron tiling_output_for_netron.onnx
```

### 示例 2: 批量转换所有 Pass 输出

```bash
# 创建输出目录
mkdir -p netron_visualization

# 批量转换
uv run python examples/fb2onnx.py --batch . netron_visualization

# 在 Netron 中查看任意文件
netron netron_visualization/tiling_output.onnx
```

## 支持的算子

转换工具支持以下 EdgeUniCompile 算子到 ONNX 的映射：

| EdgeUniCompile | ONNX |
|---------------|------|
| Conv2D | Conv |
| MaxPool2D | MaxPool |
| AveragePool2D | AveragePool |
| Relu | Relu |
| Sigmoid | Sigmoid |
| Tanh | Tanh |
| Softmax | Softmax |
| Add | Add |
| Subtract | Sub |
| Multiply | Mul |
| Divide | Div |
| MatMul | MatMul |
| Reshape | Reshape |
| Transpose | Transpose |
| Flatten | Flatten |
| Gemm | Gemm |
| BatchNorm | BatchNormalization |
| Concat | Concat |
| Split | Split |
| Pad | Pad |
| GlobalAveragePool | GlobalAveragePool |
| GlobalMaxPool | GlobalMaxPool |

## 属性转换

以下属性会被正确转换：

- `kernel_shape` → `kernel_shape`
- `strides` → `strides`
- `pads` → `pads`
- `dilations` → `dilations`
- `group/groups` → `group`
- `axis` → `axis`
- `perm` → `perm`
- `shape` → `shape`

**注意**: `tiling` 属性仅用于 EdgeUniCompile 内部优化，不会出现在 ONNX 模型中。

## 输出文件说明

转换后的 ONNX 文件包含：

1. **计算图结构**: 所有节点和连接关系
2. **张量信息**: 输入、输出和中间张量的形状和数据类型
3. **权重数据**: 如果有可用的权重数据
4. **模型元数据**: 生产者信息、域等

## 故障排除

### 错误：File not found
确保输入文件路径正确，可以使用绝对路径。

### 错误：Failed to load file
输入的 FlatBuffer 文件可能格式不正确或已损坏。检查文件是否由 EdgeUniCompile 生成。

### 错误：ONNX model validation warning
某些模型可能在 ONNX 验证时有警告，但通常仍可在 Netron 中正常查看。

### Netron 无法打开文件
1. 确保文件扩展名为 `.onnx`
2. 尝试使用 Netron 网站版 https://netron.app/
3. 检查文件是否完整（文件大小不为 0）

## 技术实现

转换工具位于 `examples/fb2onnx.py`，核心类为 `FlatBufferToONNXConverter`。

转换流程：
1. 使用 `FlatBufferBuilder.load_from_file()` 加载 FlatBuffer
2. 解析为 EdgeUniCompile Graph IR
3. 将 Graph IR 转换为 ONNX ModelProto
4. 使用 `onnx.save()` 保存为 `.onnx` 文件

## API 使用

您也可以在 Python 代码中直接使用转换功能：

```python
import edgeunicompile as euc
from examples.fb2onnx import convert_file, FlatBufferToONNXConverter

# 方法 1: 使用 convert_file 函数
convert_file("input.fb", "output.onnx")

# 方法 2: 使用 FlatBufferToONNXConverter 类
graph = euc.FlatBufferBuilder.load_from_file("input.fb")
onnx_model = FlatBufferToONNXConverter.convert(graph, "my_model")
onnx.save(onnx_model, "output.onnx")
```
