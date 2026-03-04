#include "edgeunicompile/passes/constant_folding_pass.h"
#include <cmath>
#include <cstring>

namespace edgeunic {

ConstantFoldingPass::ConstantFoldingPass() : PassBase("constant_folding_pass") {}

std::string ConstantFoldingPass::GetDescription() const {
    return "ConstantFoldingPass: Evaluates constant expressions at compile time";
}

bool ConstantFoldingPass::AreAllInputsConstant(const Node& node) const {
    for (const auto& input : node.GetInputs()) {
        if (!input || !input->IsConstant()) {
            return false;
        }
    }
    return true;
}

Status ConstantFoldingPass::FoldNode(NodePtr node, GraphPtr graph) {
    // This is a simplified implementation
    // A full implementation would need to:
    // 1. Read constant input data
    // 2. Compute the result based on op type
    // 3. Create a new constant tensor with the result
    // 4. Replace the node's output with the new constant

    OpType op_type = node->GetOpType();

    // Only handle simple element-wise operations for now
    if (op_type != OpType::kAdd && op_type != OpType::kSubtract &&
        op_type != OpType::kMultiply && op_type != OpType::kDivide) {
        return Status::NotImplemented("Constant folding not implemented for this op type");
    }

    if (node->NumInputs() < 2) {
        return Status::InvalidArgument("Binary operations require at least 2 inputs");
    }

    auto input0 = node->GetInputs()[0];
    auto input1 = node->GetInputs()[1];

    if (!input0 || !input1) {
        return Status::InvalidArgument("Invalid input tensors");
    }

    const auto& data0 = input0->GetData();
    const auto& data1 = input1->GetData();

    if (data0.empty() || data1.empty()) {
        return Status::InvalidArgument("Input tensors have no data");
    }

    // For simplicity, assume same shape and float32
    size_t num_elements = input0->NumElements();
    size_t expected_bytes = num_elements * sizeof(float);

    if (data0.size() != expected_bytes || data1.size() != expected_bytes) {
        return Status::InvalidArgument("Input tensor size mismatch");
    }

    // Create output data
    std::vector<uint8_t> result_data(expected_bytes);
    const float* ptr0 = reinterpret_cast<const float*>(data0.data());
    const float* ptr1 = reinterpret_cast<const float*>(data1.data());
    float* ptr_out = reinterpret_cast<float*>(result_data.data());

    switch (op_type) {
        case OpType::kAdd:
            for (size_t i = 0; i < num_elements; ++i) {
                ptr_out[i] = ptr0[i] + ptr1[i];
            }
            break;
        case OpType::kSubtract:
            for (size_t i = 0; i < num_elements; ++i) {
                ptr_out[i] = ptr0[i] - ptr1[i];
            }
            break;
        case OpType::kMultiply:
            for (size_t i = 0; i < num_elements; ++i) {
                ptr_out[i] = ptr0[i] * ptr1[i];
            }
            break;
        case OpType::kDivide:
            for (size_t i = 0; i < num_elements; ++i) {
                if (ptr1[i] == 0) {
                    return Status::InvalidArgument("Division by zero");
                }
                ptr_out[i] = ptr0[i] / ptr1[i];
            }
            break;
        default:
            return Status::NotImplemented("Unexpected op type");
    }

    // Create new constant tensor with the result
    auto output_tensor = node->GetOutputs()[0];
    if (output_tensor) {
        output_tensor->SetData(result_data);
        output_tensor->SetIsConstant(true);
    }

    // Note: In a full implementation, we would also remove the node
    // from the graph since its output is now a constant

    return Status::Ok();
}

Status ConstantFoldingPass::Run(GraphPtr graph, std::shared_ptr<PassContext> context) {
    if (!graph) {
        return Status::InvalidArgument("Graph cannot be null");
    }

    int folded_count = 0;
    const auto& nodes = graph->GetNodes();

    // Collect nodes to fold (can't modify while iterating)
    std::vector<NodePtr> nodes_to_fold;
    for (const auto& node : nodes) {
        if (AreAllInputsConstant(*node)) {
            nodes_to_fold.push_back(node);
        }
    }

    // Fold each node
    for (const auto& node : nodes_to_fold) {
        auto status = FoldNode(node, graph);
        if (status.IsOk()) {
            folded_count++;
        } else {
            // Log but continue - don't fail the whole pass
            // In production, you might want to log this
        }
    }

    // Update context if available
    if (context) {
        context->IncrementCounter("constants_folded", folded_count);
    }

    return Status::Ok();
}

}  // namespace edgeunic
