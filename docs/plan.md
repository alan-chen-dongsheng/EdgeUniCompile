1. python 的pass 要和 c++的pass 按照顺序一起跑
2. onnx定义一个简单的Eltwise算子,前端转为flatbuffer版本的IR
3. 编译器基于flatbuffer的版本, python的pass检查是否需要tiling,如果需要tiling则进行
4. 然后c++的pass紧接上上一个python的完成的pass,开始实现一个pass打印一下网络里目前所有node的名字
5. 在c++里写一个pass, 按照linear scan进行网络的内存分配, sram base地址是0, 最大是3M, dram基础地址是0, 最大内存是5G

## 新增实现 指令执行顺序

以下内容请使用c++实现:
网络中每一个node, 都有三个指令:
1. load [加载所有ifm , node 的 kernel等计算需要的数据]
2. exec [执行计算]
3. store [存储结果,将结果从sram写入到dram]

然后通过深度优先的拓扑排序,建立node的顺序,在根据node顺序,逐个给node建立指令,
load里面可能有多条load, 比如load ifm1 , load ifm2 ,load weight, load bias, load gamma, load beta等
exec只有一条, store 也只有一条
load指令之间是没有顺序的, 但是每一个node要先load完, 才能exec, exce完才能store,
根据node顺序, 生成的指令就这样有了依赖, 根据依赖,建立linear scan算法, 将可以不冲突,并行执行的指令, 放到一个packet里进行,
packet的含义就是说这个packet里的指令都没有依赖关系,可以并行执行. 然后packet就是指令的顺序.



