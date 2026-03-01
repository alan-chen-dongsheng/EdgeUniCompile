#include "edgeunicompile/ir/node.h"

namespace edgeunic {

Node::Node(const std::string& name, OpType op_type) : name_(name), op_type_(op_type) {}

void Node::AddInput(const TensorPtr& tensor) {
    for (const auto& existing : inputs_) {
        if (existing == tensor) {
            return;
        }
    }
    inputs_.push_back(tensor);
}

void Node::RemoveInput(const TensorPtr& tensor) {
    auto it = inputs_.begin();
    while (it != inputs_.end()) {
        if (*it == tensor) {
            it = inputs_.erase(it);
        } else {
            ++it;
        }
    }
}

void Node::AddOutput(const TensorPtr& tensor) {
    for (const auto& existing : outputs_) {
        if (existing == tensor) {
            return;
        }
    }
    outputs_.push_back(tensor);
}

void Node::RemoveOutput(const TensorPtr& tensor) {
    auto it = outputs_.begin();
    while (it != outputs_.end()) {
        if (*it == tensor) {
            it = outputs_.erase(it);
        } else {
            ++it;
        }
    }
}

void Node::SetAttribute(const std::string& name, const AttributeValue& value) {
    attributes_[name] = value;
}

std::optional<AttributeValue> Node::GetAttribute(const std::string& name) const {
    auto it = attributes_.find(name);
    if (it != attributes_.end()) {
        return it->second;
    }
    return std::nullopt;
}

bool Node::IsValid() const {
    if (name_.empty()) {
        return false;
    }
    if (op_type_ == OpType::kUnknown) {
        return false;
    }
    for (const auto& input : inputs_) {
        if (!input || !input->IsValid()) {
            return false;
        }
    }
    for (const auto& output : outputs_) {
        if (!output || !output->IsValid()) {
            return false;
        }
    }
    return true;
}

std::string Node::ToString() const {
    std::string result = name_ + ": " + OpTypeToString(op_type_);
    if (!inputs_.empty()) {
        result += "(";
        for (size_t i = 0; i < inputs_.size(); ++i) {
            if (i > 0) {
                result += ", ";
            }
            result += inputs_[i]->GetName();
        }
        result += ")";
    }
    if (!outputs_.empty()) {
        result += " -> ";
        for (size_t i = 0; i < outputs_.size(); ++i) {
            if (i > 0) {
                result += ", ";
            }
            result += outputs_[i]->GetName();
        }
    }
    return result;
}

}  // namespace edgeunic
