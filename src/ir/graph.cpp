#include "edgeunicompile/ir/graph.h"

namespace edgeunic {

void Graph::AddNode(const NodePtr& node) {
    if (node_map_.find(node->GetName()) != node_map_.end()) {
        return;
    }
    nodes_.push_back(node);
    node_map_[node->GetName()] = node;
}

void Graph::RemoveNode(const NodePtr& node) {
    auto it = nodes_.begin();
    while (it != nodes_.end()) {
        if (*it == node) {
            it = nodes_.erase(it);
        } else {
            ++it;
        }
    }
    node_map_.erase(node->GetName());
}

NodePtr Graph::GetNode(const std::string& name) const {
    auto it = node_map_.find(name);
    if (it != node_map_.end()) {
        return it->second;
    }
    return nullptr;
}

void Graph::AddTensor(const TensorPtr& tensor) {
    if (tensor_map_.find(tensor->GetName()) != tensor_map_.end()) {
        return;
    }
    tensors_.push_back(tensor);
    tensor_map_[tensor->GetName()] = tensor;
}

void Graph::RemoveTensor(const TensorPtr& tensor) {
    auto it = tensors_.begin();
    while (it != tensors_.end()) {
        if (*it == tensor) {
            it = tensors_.erase(it);
        } else {
            ++it;
        }
    }
    tensor_map_.erase(tensor->GetName());
}

TensorPtr Graph::GetTensor(const std::string& name) const {
    auto it = tensor_map_.find(name);
    if (it != tensor_map_.end()) {
        return it->second;
    }
    return nullptr;
}

TensorPtr Graph::GetInputTensor(const std::string& name) const {
    for (const auto& tensor : input_tensors_) {
        if (tensor->GetName() == name) {
            return tensor;
        }
    }
    return nullptr;
}

std::vector<TensorPtr> Graph::GetInputTensors() const {
    return input_tensors_;
}

void Graph::AddInputTensor(const TensorPtr& tensor) {
    if (!GetInputTensor(tensor->GetName())) {
        input_tensors_.push_back(tensor);
        AddTensor(tensor);
    }
}

void Graph::RemoveInputTensor(const TensorPtr& tensor) {
    auto it = input_tensors_.begin();
    while (it != input_tensors_.end()) {
        if (*it == tensor) {
            it = input_tensors_.erase(it);
        } else {
            ++it;
        }
    }
}

TensorPtr Graph::GetOutputTensor(const std::string& name) const {
    for (const auto& tensor : output_tensors_) {
        if (tensor->GetName() == name) {
            return tensor;
        }
    }
    return nullptr;
}

std::vector<TensorPtr> Graph::GetOutputTensors() const {
    return output_tensors_;
}

void Graph::AddOutputTensor(const TensorPtr& tensor) {
    if (!GetOutputTensor(tensor->GetName())) {
        output_tensors_.push_back(tensor);
        AddTensor(tensor);
    }
}

void Graph::RemoveOutputTensor(const TensorPtr& tensor) {
    auto it = output_tensors_.begin();
    while (it != output_tensors_.end()) {
        if (*it == tensor) {
            it = output_tensors_.erase(it);
        } else {
            ++it;
        }
    }
}

std::vector<NodePtr> Graph::GetTopologicalOrder() const {
    std::unordered_map<std::string, int> in_degree;
    std::unordered_map<std::string, std::vector<std::string>> adj;

    for (const auto& node : nodes_) {
        in_degree[node->GetName()] = 0;
        adj[node->GetName()] = {};
    }

    for (const auto& u : nodes_) {
        for (const auto& output : u->GetOutputs()) {
            for (const auto& v : nodes_) {
                if (u != v) {
                    for (const auto& input : v->GetInputs()) {
                        if (input == output) {
                            adj[u->GetName()].push_back(v->GetName());
                            in_degree[v->GetName()]++;
                        }
                    }
                }
            }
        }
    }

    std::vector<std::string> queue;
    for (const auto& [name, degree] : in_degree) {
        if (degree == 0) {
            queue.push_back(name);
        }
    }

    std::vector<NodePtr> topological;
    while (!queue.empty()) {
        std::string u = queue.front();
        queue.erase(queue.begin());
        topological.push_back(node_map_.at(u));

        for (const std::string& v : adj.at(u)) {
            if (--in_degree.at(v) == 0) {
                queue.push_back(v);
            }
        }
    }

    return topological;
}

Status Graph::IsValidImpl() const {
    for (const auto& node : nodes_) {
        if (!node->IsValid()) {
            return Status::Error("Invalid node: " + node->GetName());
        }
    }

    for (const auto& tensor : tensors_) {
        if (!tensor->IsValid()) {
            return Status::Error("Invalid tensor: " + tensor->GetName());
        }
    }

    for (const auto& tensor : input_tensors_) {
        if (!GetTensor(tensor->GetName())) {
            return Status::Error("Input tensor not found in graph: " + tensor->GetName());
        }
    }

    for (const auto& tensor : output_tensors_) {
        if (!GetTensor(tensor->GetName())) {
            return Status::Error("Output tensor not found in graph: " + tensor->GetName());
        }
    }

    return Status::Ok();
}

bool Graph::IsValid() const {
    Status status = IsValidImpl();
    return status.IsOk();
}

std::string Graph::ToString() const {
    std::string result = "Graph: " + name_ + "\n";

    if (!input_tensors_.empty()) {
        result += "Inputs: [";
        for (size_t i = 0; i < input_tensors_.size(); ++i) {
            if (i > 0) {
                result += ", ";
            }
            result += input_tensors_[i]->GetName();
        }
        result += "]\n";
    }

    if (!output_tensors_.empty()) {
        result += "Outputs: [";
        for (size_t i = 0; i < output_tensors_.size(); ++i) {
            if (i > 0) {
                result += ", ";
            }
            result += output_tensors_[i]->GetName();
        }
        result += "]\n";
    }

    if (!nodes_.empty()) {
        result += "Nodes:\n";
        for (const auto& node : nodes_) {
            result += "  " + node->ToString() + "\n";
        }
    }

    if (!tensors_.empty()) {
        result += "Tensors:\n";
        for (const auto& tensor : tensors_) {
            result += "  " + tensor->ToString() + "\n";
        }
    }

    return result;
}

}  // namespace edgeunic
