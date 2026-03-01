#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include "edgeunicompile/core/types.h"
#include "tensor.h"

namespace edgeunic {

class Node {
public:
    Node() = default;
    Node(const std::string& name, OpType op_type);

    std::string GetName() const { return name_; }
    void SetName(const std::string& name) { name_ = name; }

    OpType GetOpType() const { return op_type_; }
    void SetOpType(OpType op_type) { op_type_ = op_type; }

    void AddInput(const TensorPtr& tensor);
    void RemoveInput(const TensorPtr& tensor);
    const std::vector<TensorPtr>& GetInputs() const { return inputs_; }

    void AddOutput(const TensorPtr& tensor);
    void RemoveOutput(const TensorPtr& tensor);
    const std::vector<TensorPtr>& GetOutputs() const { return outputs_; }

    void SetAttribute(const std::string& name, const AttributeValue& value);
    std::optional<AttributeValue> GetAttribute(const std::string& name) const;
    const std::unordered_map<std::string, AttributeValue>& GetAttributes() const { return attributes_; }

    size_t NumInputs() const { return inputs_.size(); }
    size_t NumOutputs() const { return outputs_.size(); }

    bool IsValid() const;
    std::string ToString() const;

private:
    std::string name_;
    OpType op_type_ = OpType::kUnknown;
    std::vector<TensorPtr> inputs_;
    std::vector<TensorPtr> outputs_;
    std::unordered_map<std::string, AttributeValue> attributes_;
};

using NodePtr = std::shared_ptr<Node>;

}  // namespace edgeunic
