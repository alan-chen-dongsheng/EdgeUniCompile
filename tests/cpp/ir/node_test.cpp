#include <gtest/gtest.h>
#include "edgeunicompile/ir/node.h"
#include "edgeunicompile/ir/tensor.h"

using namespace edgeunic;

TEST(NodeTest, CreateNode) {
    Node node("test_node", OpType::kAdd);
    EXPECT_EQ(node.GetName(), "test_node");
    EXPECT_EQ(node.GetOpType(), OpType::kAdd);
    EXPECT_TRUE(node.GetInputs().empty());
    EXPECT_TRUE(node.GetOutputs().empty());
}

TEST(NodeTest, SetAndGetName) {
    Node node("test_node", OpType::kAdd);
    EXPECT_EQ(node.GetName(), "test_node");

    node.SetName("new_name");
    EXPECT_EQ(node.GetName(), "new_name");
}

TEST(NodeTest, SetAndGetOpType) {
    Node node("test_node", OpType::kAdd);
    EXPECT_EQ(node.GetOpType(), OpType::kAdd);

    node.SetOpType(OpType::kConv2D);
    EXPECT_EQ(node.GetOpType(), OpType::kConv2D);
}

TEST(NodeTest, AddInputs) {
    Node node("test_node", OpType::kAdd);

    TensorPtr tensor1 = std::make_shared<Tensor>("tensor1", DataType::kFloat32, Shape{{2, 3}});
    TensorPtr tensor2 = std::make_shared<Tensor>("tensor2", DataType::kFloat32, Shape{{2, 3}});

    node.AddInput(tensor1);
    node.AddInput(tensor2);

    EXPECT_TRUE(node.GetInputs().size() == 2);
    EXPECT_TRUE(node.GetInputs()[0] == tensor1);
    EXPECT_TRUE(node.GetInputs()[1] == tensor2);
}

TEST(NodeTest, RemoveInputs) {
    Node node("test_node", OpType::kAdd);

    TensorPtr tensor1 = std::make_shared<Tensor>("tensor1", DataType::kFloat32, Shape{{2, 3}});
    TensorPtr tensor2 = std::make_shared<Tensor>("tensor2", DataType::kFloat32, Shape{{2, 3}});

    node.AddInput(tensor1);
    node.AddInput(tensor2);

    node.RemoveInput(tensor1);
    EXPECT_TRUE(node.GetInputs().size() == 1);
    EXPECT_TRUE(node.GetInputs()[0] == tensor2);
}

TEST(NodeTest, AddOutputs) {
    Node node("test_node", OpType::kAdd);

    TensorPtr tensor = std::make_shared<Tensor>("output", DataType::kFloat32, Shape{{2, 3}});
    node.AddOutput(tensor);

    EXPECT_TRUE(node.GetOutputs().size() == 1);
    EXPECT_TRUE(node.GetOutputs()[0] == tensor);
}

TEST(NodeTest, RemoveOutputs) {
    Node node("test_node", OpType::kAdd);

    TensorPtr tensor = std::make_shared<Tensor>("output", DataType::kFloat32, Shape{{2, 3}});
    node.AddOutput(tensor);
    node.RemoveOutput(tensor);

    EXPECT_TRUE(node.GetOutputs().empty());
}

TEST(NodeTest, AddAndGetAttributes) {
    Node node("test_node", OpType::kAdd);

    node.SetAttribute("key1", int64_t(42));
    EXPECT_EQ(std::get<int64_t>(node.GetAttribute("key1").value()), int64_t(42));

    node.SetAttribute("key2", 3.14f);
    EXPECT_EQ(std::get<float>(node.GetAttribute("key2").value()), 3.14f);
}

TEST(NodeTest, IsValid) {
    Node valid_node("test_node", OpType::kAdd);
    valid_node.AddInput(std::make_shared<Tensor>("input", DataType::kFloat32, Shape{{2, 3}}));
    valid_node.AddOutput(std::make_shared<Tensor>("output", DataType::kFloat32, Shape{{2, 3}}));
    EXPECT_TRUE(valid_node.IsValid());

    Node invalid_node("", OpType::kAdd);
    EXPECT_FALSE(invalid_node.IsValid());

    Node invalid_op_node("invalid_node", OpType::kUnknown);
    EXPECT_FALSE(invalid_op_node.IsValid());
}
