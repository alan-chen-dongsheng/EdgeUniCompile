1. python 的pass 要和 c++的pass 按照顺序一起跑
2. onnx定义一个简单的Eltwise算子,前端转为flatbuffer版本的IR
3. 编译器基于flatbuffer的版本, python的pass检查是否需要tiling,如果需要tiling则进行
4. 然后c++的pass紧接上上一个python的完成的pass,开始实现一个pass打印一下网络里目前所有node的名字
5. 在c++里写一个pass, 按照linear scan进行网络的内存分配, sram base地址是0, 最大是3M, dram基础地址是0, 最大内存是5G

