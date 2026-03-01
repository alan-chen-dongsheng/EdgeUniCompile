#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include "edgeunicompile/core/types.h"
#include "node.h"
#include "tensor.h"

namespace edgeunic {

class Graph {
public:
    Graph() = default;
    explicit Graph(const std::string& name) : name_(name) {}

    std::string GetName() const { return name_; }
    void SetName(const std::string& name) { name_ = name; }

    void AddNode(const NodePtr& node);
    void RemoveNode(const NodePtr& node);
    NodePtr GetNode(const std::string& name) const;
    const std::vector<NodePtr>& GetNodes() const { return nodes_; }

    void AddTensor(const TensorPtr& tensor);
    void RemoveTensor(const TensorPtr& tensor);
    TensorPtr GetTensor(const std::string& name) const;
    const std::vector<TensorPtr>& GetTensors() const { return tensors_; }

    TensorPtr GetInputTensor(const std::string& name) const;
    std::vector<TensorPtr> GetInputTensors() const;
    void AddInputTensor(const TensorPtr& tensor);
    void RemoveInputTensor(const TensorPtr& tensor);

    TensorPtr GetOutputTensor(const std::string& name) const;
    std::vector<TensorPtr> GetOutputTensors() const;
    void AddOutputTensor(const TensorPtr& tensor);
    void RemoveOutputTensor(const TensorPtr& tensor);

    std::vector<NodePtr> GetTopologicalOrder() const;
    bool IsValid() const;

    size_t NumNodes() const { return nodes_.size(); }
    size_t NumTensors() const { return tensors_.size(); }

    std::string ToString() const;

private:
    Status IsValidImpl() const;

    std::string name_;
    std::vector<NodePtr> nodes_;
    std::vector<TensorPtr> tensors_;
    std::vector<TensorPtr> input_tensors_;
    std::vector<TensorPtr> output_tensors_;

    std::unordered_map<std::string, NodePtr> node_map_;
    std::unordered_map<std::string, TensorPtr> tensor_map_;
};

using GraphPtr = std::shared_ptr<Graph>;

}  // namespace edgeunic
