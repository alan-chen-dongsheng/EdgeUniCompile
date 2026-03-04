#include <gtest/gtest.h>
#include "edgeunicompile/codegen/instruction.h"
#include "edgeunicompile/codegen/instruction_scheduler.h"
#include "edgeunicompile/ir/graph.h"
#include "edgeunicompile/ir/tensor.h"
#include "edgeunicompile/ir/node.h"
#include "edgeunicompile/core/context.h"

using namespace edgeunic;

namespace {

// Helper function to create a tensor
TensorPtr CreateTensor(const std::string& name, const Shape& shape) {
    auto tensor = std::make_shared<Tensor>(name, DataType::kFloat32, shape);
    return tensor;
}

// Helper function to create a node with string op type (convenience)
NodePtr CreateNodeWithStringOp(const std::string& name, const std::string& op_type_str) {
    auto node = std::make_shared<Node>();
    node->SetName(name);
    node->SetOpType(StringToOpType(op_type_str));
    return node;
}

}  // namespace

/**
 * Test Instruction creation and basic properties.
 */
TEST(InstructionTest, CreateLoadInstruction) {
    auto load = std::make_shared<LoadInstruction>();
    load->SetName("test_load");
    load->SetNodeId(0);
    load->SetTensorName("input_tensor");
    load->SetSourceDramAddress(0x1000);
    load->SetTargetSramAddress(0x100);
    load->SetSizeBytes(1024);

    EXPECT_EQ(load->GetType(), InstructionType::kLoad);
    EXPECT_EQ(load->GetName(), "test_load");
    EXPECT_EQ(load->GetNodeId(), 0);
    EXPECT_EQ(load->GetTensorName(), "input_tensor");
    EXPECT_EQ(load->GetSourceDramAddress(), static_cast<uint64_t>(0x1000));
    EXPECT_EQ(load->GetTargetSramAddress(), static_cast<uint64_t>(0x100));
    EXPECT_EQ(load->GetSizeBytes(), 1024u);

    // Test string conversion
    std::string str = load->ToString();
    EXPECT_FALSE(str.empty());
    EXPECT_NE(str.find("LOAD"), std::string::npos);
}

TEST(InstructionTest, CreateExecInstruction) {
    auto exec = std::make_shared<ExecInstruction>();
    exec->SetName("test_exec");
    exec->SetNodeId(1);
    exec->SetOpType(OpType::kConv2D);
    exec->SetOutputSramAddress(0x200);
    exec->AddInputSramAddress(0x100);
    exec->AddInputSramAddress(0x150);

    EXPECT_EQ(exec->GetType(), InstructionType::kExec);
    EXPECT_EQ(exec->GetOpType(), OpType::kConv2D);
    EXPECT_EQ(exec->GetOutputSramAddress(), static_cast<uint64_t>(0x200));
    EXPECT_EQ(exec->GetInputSramAddresses().size(), 2u);

    // Test string conversion
    std::string str = exec->ToString();
    EXPECT_FALSE(str.empty());
    EXPECT_NE(str.find("EXEC"), std::string::npos);
    EXPECT_NE(str.find("Conv2D"), std::string::npos);
}

TEST(InstructionTest, CreateStoreInstruction) {
    auto store = std::make_shared<StoreInstruction>();
    store->SetName("test_store");
    store->SetNodeId(2);
    store->SetTensorName("output_tensor");
    store->SetSourceSramAddress(0x200);
    store->SetTargetDramAddress(0x5000);
    store->SetSizeBytes(2048);

    EXPECT_EQ(store->GetType(), InstructionType::kStore);
    EXPECT_EQ(store->GetTensorName(), "output_tensor");
    EXPECT_EQ(store->GetSourceSramAddress(), static_cast<uint64_t>(0x200));
    EXPECT_EQ(store->GetTargetDramAddress(), static_cast<uint64_t>(0x5000));
    EXPECT_EQ(store->GetSizeBytes(), 2048u);

    // Test string conversion
    std::string str = store->ToString();
    EXPECT_FALSE(str.empty());
    EXPECT_NE(str.find("STORE"), std::string::npos);
}

/**
 * Test InstructionPacket functionality.
 */
TEST(InstructionPacketTest, CreatePacket) {
    InstructionPacket packet;

    auto load1 = std::make_shared<LoadInstruction>();
    load1->SetName("load1");
    load1->SetNodeId(0);

    auto load2 = std::make_shared<LoadInstruction>();
    load2->SetName("load2");
    load2->SetNodeId(0);

    packet.AddInstruction(load1);
    packet.AddInstruction(load2);

    EXPECT_EQ(packet.NumInstructions(), 2u);
    EXPECT_EQ(packet.GetInstructions().size(), 2u);

    // Test string conversion
    std::string str = packet.ToString();
    EXPECT_FALSE(str.empty());
    EXPECT_NE(str.find("PACKET"), std::string::npos);
}

/**
 * Test InstructionScheduler with a simple graph.
 */
TEST(InstructionSchedulerTest, CreateScheduler) {
    InstructionScheduler scheduler;

    // Create a simple graph: Input -> Conv2D -> Output
    auto graph = std::make_shared<Graph>("test_graph");

    // Create tensors
    auto input = CreateTensor("input", Shape{1, 3, 32, 32});      // NCHW: 1*3*32*32*4 = 12288 bytes
    auto weight = CreateTensor("weight", Shape{16, 3, 3, 3});     // 16*3*3*3*4 = 1728 bytes
    auto bias = CreateTensor("bias", Shape{16});                   // 16*4 = 64 bytes
    auto output = CreateTensor("output", Shape{1, 16, 30, 30});   // 1*16*30*30*4 = 57600 bytes

    graph->AddTensor(input);
    graph->AddTensor(weight);
    graph->AddTensor(bias);
    graph->AddTensor(output);

    // Create Conv2D node
    auto conv_node = CreateNodeWithStringOp("conv1", "Conv2D");
    conv_node->AddInput(input);
    conv_node->AddInput(weight);
    conv_node->AddInput(bias);
    conv_node->AddOutput(output);
    conv_node->SetAttribute("kernel_shape", std::vector<int64_t>{3, 3});
    conv_node->SetAttribute("strides", std::vector<int64_t>{1, 1});
    conv_node->SetAttribute("pads", std::vector<int64_t>{0, 0, 0, 0});

    graph->AddNode(conv_node);
    graph->AddInputTensor(input);
    graph->AddOutputTensor(output);

    // Generate instructions
    auto context = std::make_shared<PassContext>();
    auto status = scheduler.GenerateInstructions(graph, context);

    EXPECT_TRUE(status.IsOk());
    EXPECT_FALSE(scheduler.GetInstructions().empty());
    EXPECT_FALSE(scheduler.GetNodeOrder().empty());

    // Should have: 3 LOADs (input, weight, bias) + 1 EXEC + 1 STORE = 5 instructions
    EXPECT_EQ(scheduler.GetInstructions().size(), 5u);

    // Verify instruction types
    const auto& instructions = scheduler.GetInstructions();
    int load_count = 0, exec_count = 0, store_count = 0;
    for (const auto& instr : instructions) {
        switch (instr->GetType()) {
            case InstructionType::kLoad:
                load_count++;
                break;
            case InstructionType::kExec:
                exec_count++;
                break;
            case InstructionType::kStore:
                store_count++;
                break;
        }
    }
    EXPECT_EQ(load_count, 3);
    EXPECT_EQ(exec_count, 1);
    EXPECT_EQ(store_count, 1);
}

/**
 * Test InstructionScheduler with multi-node graph.
 */
TEST(InstructionSchedulerTest, MultiNodeGraph) {
    InstructionScheduler scheduler;

    // Create a graph with two Conv2D nodes in sequence
    auto graph = std::make_shared<Graph>("multi_node_graph");

    // Create tensors
    auto input = CreateTensor("input", Shape{1, 3, 32, 32});
    auto weight1 = CreateTensor("weight1", Shape{16, 3, 3, 3});
    auto bias1 = CreateTensor("bias1", Shape{16});
    auto output1 = CreateTensor("output1", Shape{1, 16, 30, 30});
    auto weight2 = CreateTensor("weight2", Shape{32, 16, 3, 3});
    auto bias2 = CreateTensor("bias2", Shape{32});
    auto output2 = CreateTensor("output2", Shape{1, 32, 28, 28});

    graph->AddTensor(input);
    graph->AddTensor(weight1);
    graph->AddTensor(bias1);
    graph->AddTensor(output1);
    graph->AddTensor(weight2);
    graph->AddTensor(bias2);
    graph->AddTensor(output2);

    // Create first Conv2D node
    auto conv1 = CreateNodeWithStringOp("conv1", "Conv2D");
    conv1->AddInput(input);
    conv1->AddInput(weight1);
    conv1->AddInput(bias1);
    conv1->AddOutput(output1);

    // Create second Conv2D node (takes output1 as input)
    auto conv2 = CreateNodeWithStringOp("conv2", "Conv2D");
    conv2->AddInput(output1);
    conv2->AddInput(weight2);
    conv2->AddInput(bias2);
    conv2->AddOutput(output2);

    graph->AddNode(conv1);
    graph->AddNode(conv2);
    graph->AddInputTensor(input);
    graph->AddOutputTensor(output2);

    // Generate instructions
    auto status = scheduler.GenerateInstructions(graph);
    EXPECT_TRUE(status.IsOk());

    // Verify node order (conv1 should come before conv2)
    const auto& node_order = scheduler.GetNodeOrder();
    EXPECT_EQ(node_order.size(), 2u);

    size_t conv1_idx = 0, conv2_idx = 0;
    for (size_t i = 0; i < node_order.size(); ++i) {
        if (node_order[i] == "conv1") conv1_idx = i;
        if (node_order[i] == "conv2") conv2_idx = i;
    }
    EXPECT_LT(conv1_idx, conv2_idx);

    // Schedule instructions
    status = scheduler.ScheduleInstructions();
    EXPECT_TRUE(status.IsOk());

    // Verify packets
    const auto& packets = scheduler.GetPackets();
    EXPECT_FALSE(packets.empty());

    // Print schedule for debugging
    scheduler.PrintSchedule();
}

/**
 * Test InstructionScheduler with Eltwise operation.
 */
TEST(InstructionSchedulerTest, EltwiseGraph) {
    InstructionScheduler scheduler;

    // Create a graph with Eltwise operation (e.g., Relu)
    auto graph = std::make_shared<Graph>("eltwise_graph");

    // Create tensors
    auto input = CreateTensor("input", Shape{1, 16, 32, 32});
    auto output = CreateTensor("output", Shape{1, 16, 32, 32});

    graph->AddTensor(input);
    graph->AddTensor(output);

    // Create Relu node
    auto relu = CreateNodeWithStringOp("relu1", "Relu");
    relu->AddInput(input);
    relu->AddOutput(output);

    graph->AddNode(relu);
    graph->AddInputTensor(input);
    graph->AddOutputTensor(output);

    // Generate instructions
    auto status = scheduler.GenerateInstructions(graph);
    EXPECT_TRUE(status.IsOk());

    // Should have: 1 LOAD + 1 EXEC + 1 STORE = 3 instructions
    EXPECT_EQ(scheduler.GetInstructions().size(), 3u);

    // Verify instruction order
    const auto& instructions = scheduler.GetInstructions();
    EXPECT_EQ(instructions[0]->GetType(), InstructionType::kLoad);
    EXPECT_EQ(instructions[1]->GetType(), InstructionType::kExec);
    EXPECT_EQ(instructions[2]->GetType(), InstructionType::kStore);

    // Verify EXEC operation type
    auto exec = std::static_pointer_cast<ExecInstruction>(instructions[1]);
    EXPECT_EQ(exec->GetOpType(), OpType::kRelu);
}

/**
 * Test InstructionScheduler dependencies.
 */
TEST(InstructionSchedulerTest, InstructionDependencies) {
    InstructionScheduler scheduler;

    // Create a simple graph
    auto graph = std::make_shared<Graph>("dep_test_graph");

    auto input = CreateTensor("input", Shape{1, 3, 32, 32});
    auto weight = CreateTensor("weight", Shape{16, 3, 3, 3});
    auto output = CreateTensor("output", Shape{1, 16, 30, 30});

    graph->AddTensor(input);
    graph->AddTensor(weight);
    graph->AddTensor(output);

    auto conv = CreateNodeWithStringOp("conv1", "Conv2D");
    conv->AddInput(input);
    conv->AddInput(weight);
    conv->AddOutput(output);

    graph->AddNode(conv);
    graph->AddInputTensor(input);
    graph->AddOutputTensor(output);

    // Generate and schedule instructions
    auto status = scheduler.GenerateInstructions(graph);
    EXPECT_TRUE(status.IsOk());

    status = scheduler.ScheduleInstructions();
    EXPECT_TRUE(status.IsOk());

    // Verify that packets respect dependencies
    const auto& packets = scheduler.GetPackets();

    // Find which packet contains each instruction type
    int load_packet_idx = -1, exec_packet_idx = -1, store_packet_idx = -1;

    for (size_t i = 0; i < packets.size(); ++i) {
        for (const auto& instr : packets[i]->GetInstructions()) {
            switch (instr->GetType()) {
                case InstructionType::kLoad:
                    if (load_packet_idx < 0) load_packet_idx = static_cast<int>(i);
                    break;
                case InstructionType::kExec:
                    if (exec_packet_idx < 0) exec_packet_idx = static_cast<int>(i);
                    break;
                case InstructionType::kStore:
                    if (store_packet_idx < 0) store_packet_idx = static_cast<int>(i);
                    break;
            }
        }
    }

    // LOAD should come before EXEC, EXEC should come before STORE
    EXPECT_LT(load_packet_idx, exec_packet_idx);
    EXPECT_LT(exec_packet_idx, store_packet_idx);
}
