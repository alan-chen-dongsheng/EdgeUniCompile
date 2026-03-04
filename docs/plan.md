1. 使用python的onnx库创建一个单卷积的onnx模型, 内存占用 > 3M
2. 前端将该模型转为仓库内定义好的FlatBuffer格式
3. 编译器接受前端FlatBuffer格式的前端IR
4. 使用cpp和python的pass,导入为编译器内的数据结构, 根据Sram 3M大小的限制, 对该IR进行tiling
5. 支持每一个Pass执行完成后,都到处IR (FlatBuffer格式)
6. 再将FlatBuffer格式的tiling后的IR, 转为 MLIR 的方言, 选择合适的MLIR内建方言.
7. 导出MLIR方言
8. 结束
