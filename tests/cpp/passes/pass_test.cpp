#include <gtest/gtest.h>
#include "edgeunicompile/core/types.h"
#include "edgeunicompile/core/context.h"
#include "edgeunicompile/ir/graph.h"
#include "edgeunicompile/ir/node.h"
#include "edgeunicompile/ir/tensor.h"
#include "edgeunicompile/passes/pass_base.h"
#include "edgeunicompile/passes/pass_manager.h"
#include "edgeunicompile/passes/constant_folding_pass.h"

using namespace edgeunic;

namespace {

TEST(PassBaseTest, CreatePass) {
    auto pass = std::make_unique<ConstantFoldingPass>();
    EXPECT_EQ(pass->GetName(), "constant_folding_pass");
    EXPECT_TRUE(pass->IsEnabled());
    pass->SetEnabled(false);
    EXPECT_FALSE(pass->IsEnabled());
}

TEST(PassBaseTest, PassDescription) {
    ConstantFoldingPass pass;
    std::string desc = pass.GetDescription();
    EXPECT_FALSE(desc.empty());
    EXPECT_NE(desc, "constant_folding_pass");  // Should be more descriptive
}

TEST(PassContextTest, Config) {
    PassContext context;

    // Test int config
    context.SetConfig("tile_size", int64_t(64));
    auto value = context.GetConfig("tile_size");
    EXPECT_TRUE(value.has_value());
    EXPECT_EQ(std::get<int64_t>(*value), 64);

    // Test float config
    context.SetConfig("threshold", 0.5f);
    auto float_val = context.GetConfig("threshold");
    EXPECT_TRUE(float_val.has_value());
    EXPECT_FLOAT_EQ(std::get<float>(*float_val), 0.5f);
}

TEST(PassContextTest, Counters) {
    PassContext context;

    EXPECT_EQ(context.GetCounter("test"), 0);

    context.IncrementCounter("test", 5);
    EXPECT_EQ(context.GetCounter("test"), 5);

    context.IncrementCounter("test", 3);
    EXPECT_EQ(context.GetCounter("test"), 8);
}

TEST(PassManagerTest, CreateAndAddPass) {
    auto context = std::make_shared<Context>();
    PassManager manager(context);

    auto pass = std::make_shared<ConstantFoldingPass>();
    manager.AddPass(pass);

    auto retrieved = manager.GetPass("constant_folding_pass");
    EXPECT_EQ(retrieved, pass);

    auto passes = manager.ListPasses();
    EXPECT_EQ(passes.size(), 1u);
    EXPECT_EQ(passes[0].first, "constant_folding_pass");
    EXPECT_TRUE(passes[0].second);
}

TEST(PassManagerTest, RemovePass) {
    PassManager manager;

    auto pass = std::make_shared<ConstantFoldingPass>();
    manager.AddPass(pass);
    EXPECT_NE(manager.GetPass("constant_folding_pass"), nullptr);

    manager.RemovePass("constant_folding_pass");
    EXPECT_EQ(manager.GetPass("constant_folding_pass"), nullptr);
}

TEST(PassManagerTest, EnableDisablePass) {
    PassManager manager;
    manager.AddPass(std::make_shared<ConstantFoldingPass>());

    manager.DisablePass("constant_folding_pass");
    auto pass = manager.GetPass("constant_folding_pass");
    EXPECT_FALSE(pass->IsEnabled());

    manager.EnablePass("constant_folding_pass");
    EXPECT_TRUE(pass->IsEnabled());
}

TEST(PassManagerTest, RunPasses) {
    auto context = std::make_shared<Context>();
    PassManager manager(context);
    manager.AddPass(std::make_shared<ConstantFoldingPass>());

    // Create a simple graph
    auto graph = std::make_shared<Graph>("test_graph");

    // Create constant tensors for add operation
    auto tensor_a = std::make_shared<Tensor>("a", DataType::kFloat32, Shape({2, 2}));
    auto tensor_b = std::make_shared<Tensor>("b", DataType::kFloat32, Shape({2, 2}));
    auto tensor_out = std::make_shared<Tensor>("out", DataType::kFloat32, Shape({2, 2}));

    // Set constant data
    std::vector<float> data_a = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> data_b = {5.0f, 6.0f, 7.0f, 8.0f};

    tensor_a->SetData(std::vector<uint8_t>(data_a.size() * sizeof(float)));
    std::memcpy(tensor_a->GetData().data(), data_a.data(), data_a.size() * sizeof(float));
    tensor_a->SetIsConstant(true);

    tensor_b->SetData(std::vector<uint8_t>(data_b.size() * sizeof(float)));
    std::memcpy(tensor_b->GetData().data(), data_b.data(), data_b.size() * sizeof(float));
    tensor_b->SetIsConstant(true);

    graph->AddTensor(tensor_a);
    graph->AddTensor(tensor_b);
    graph->AddTensor(tensor_out);

    // Create add node
    auto add_node = std::make_shared<Node>("add", OpType::kAdd);
    add_node->AddInput(tensor_a);
    add_node->AddInput(tensor_b);
    add_node->AddOutput(tensor_out);
    graph->AddNode(add_node);

    // Run passes
    auto status = manager.Run(graph);
    EXPECT_TRUE(status.IsOk());
}

TEST(ConstantFoldingPassTest, FoldAddOperation) {
    // Create graph with constant add operation
    auto graph = std::make_shared<Graph>("fold_test");
    auto context = std::make_shared<PassContext>();

    // Create tensors
    auto tensor_a = std::make_shared<Tensor>("a", DataType::kFloat32, Shape({2}));
    auto tensor_b = std::make_shared<Tensor>("b", DataType::kFloat32, Shape({2}));
    auto tensor_out = std::make_shared<Tensor>("out", DataType::kFloat32, Shape({2}));

    // Set data
    std::vector<float> data_a = {1.0f, 2.0f};
    std::vector<float> data_b = {3.0f, 4.0f};

    tensor_a->SetData(std::vector<uint8_t>(data_a.size() * sizeof(float)));
    std::memcpy(tensor_a->GetData().data(), data_a.data(), data_a.size() * sizeof(float));
    tensor_a->SetIsConstant(true);

    tensor_b->SetData(std::vector<uint8_t>(data_b.size() * sizeof(float)));
    std::memcpy(tensor_b->GetData().data(), data_b.data(), data_b.size() * sizeof(float));
    tensor_b->SetIsConstant(true);

    graph->AddTensor(tensor_a);
    graph->AddTensor(tensor_b);
    graph->AddTensor(tensor_out);

    // Create node
    auto add_node = std::make_shared<Node>("add", OpType::kAdd);
    add_node->AddInput(tensor_a);
    add_node->AddInput(tensor_b);
    add_node->AddOutput(tensor_out);
    graph->AddNode(add_node);

    // Run pass
    ConstantFoldingPass pass;
    auto status = pass.Run(graph, context);

    EXPECT_TRUE(status.IsOk());
    EXPECT_TRUE(tensor_out->IsConstant());

    // Verify result
    const auto& result = tensor_out->GetData();
    const float* result_ptr = reinterpret_cast<const float*>(result.data());
    EXPECT_FLOAT_EQ(result_ptr[0], 4.0f);  // 1 + 3
    EXPECT_FLOAT_EQ(result_ptr[1], 6.0f);  // 2 + 4
}

}  // namespace
