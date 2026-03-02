#include <gtest/gtest.h>
#include "edgeunicompile/ir/graph.h"
#include "edgeunicompile/ir/node.h"
#include "edgeunicompile/ir/tensor.h"

using namespace edgeunic;

TEST(GraphTest, CreateGraph) {
    Graph graph("test_graph");
    EXPECT_EQ(graph.GetName(), "test_graph");
    EXPECT_TRUE(graph.NumNodes() == 0);
    EXPECT_TRUE(graph.NumTensors() == 0);
    EXPECT_TRUE(graph.GetInputTensors().empty());
    EXPECT_TRUE(graph.GetOutputTensors().empty());
}

TEST(GraphTest, AddAndGetNodes) {
    Graph graph("test_graph");
    NodePtr node = std::make_shared<Node>("node1", OpType::kAdd);
    graph.AddNode(node);

    EXPECT_TRUE(graph.NumNodes() == 1u);
    EXPECT_TRUE(graph.GetNode("node1") == node);
    EXPECT_TRUE(graph.GetNode("nonexistent") == nullptr);
}

TEST(GraphTest, AddAndGetTensors) {
    Graph graph("test_graph");
    TensorPtr tensor = std::make_shared<Tensor>("tensor1", DataType::kFloat32, Shape{{2, 3}});
    graph.AddTensor(tensor);

    EXPECT_TRUE(graph.NumTensors() == 1u);
    EXPECT_TRUE(graph.GetTensor("tensor1") == tensor);
    EXPECT_TRUE(graph.GetTensor("nonexistent") == nullptr);
}

TEST(GraphTest, AddAndRemoveNodes) {
    Graph graph("test_graph");
    NodePtr node = std::make_shared<Node>("node1", OpType::kAdd);
    graph.AddNode(node);
    EXPECT_TRUE(graph.NumNodes() == 1u);

    graph.RemoveNode(node);
    EXPECT_TRUE(graph.NumNodes() == 0u);
}

TEST(GraphTest, AddAndRemoveTensors) {
    Graph graph("test_graph");
    TensorPtr tensor = std::make_shared<Tensor>("tensor1", DataType::kFloat32, Shape{{2, 3}});
    graph.AddTensor(tensor);
    EXPECT_TRUE(graph.NumTensors() == 1u);

    graph.RemoveTensor(tensor);
    EXPECT_TRUE(graph.NumTensors() == 0u);
}

TEST(GraphTest, AddAndGetInputs) {
    Graph graph("test_graph");
    TensorPtr tensor = std::make_shared<Tensor>("input1", DataType::kFloat32, Shape{{2, 3}});
    graph.AddInputTensor(tensor);

    EXPECT_FALSE(graph.GetInputTensors().empty());
    EXPECT_EQ(graph.GetInputTensors().size(), 1u);
    EXPECT_EQ(graph.GetInputTensors()[0], tensor);
}

TEST(GraphTest, AddAndGetOutputs) {
    Graph graph("test_graph");
    TensorPtr tensor = std::make_shared<Tensor>("output1", DataType::kFloat32, Shape{{2, 3}});
    graph.AddOutputTensor(tensor);

    EXPECT_FALSE(graph.GetOutputTensors().empty());
    EXPECT_EQ(graph.GetOutputTensors().size(), 1u);
    EXPECT_EQ(graph.GetOutputTensors()[0], tensor);
}
