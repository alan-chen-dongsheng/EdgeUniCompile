#include "edgeunicompile/passes/print_node_names_pass.h"
#include <iostream>
#include <sstream>

namespace edgeunic {

PrintNodeNamesPass::PrintNodeNamesPass(bool verbose)
    : PassBase("print_node_names_pass"), verbose_(verbose) {}

std::string PrintNodeNamesPass::GetDescription() const {
    return "PrintNodeNamesPass: Prints all node names in the computation graph";
}

Status PrintNodeNamesPass::Run(GraphPtr graph, std::shared_ptr<PassContext> context) {
    if (!graph) {
        return Status::InvalidArgument("Graph cannot be null");
    }

    std::ostringstream oss;
    oss << "\n" << std::string(60, '=') << "\n";
    oss << "Node Names in Graph: " << graph->GetName() << "\n";
    oss << std::string(60, '-') << "\n";

    const auto& nodes = graph->GetNodes();
    if (nodes.empty()) {
        oss << "  (No nodes in graph)\n";
    } else {
        oss << "  Total nodes: " << nodes.size() << "\n\n";
        for (size_t i = 0; i < nodes.size(); ++i) {
            const auto& node = nodes[i];
            oss << "  [" << i << "] " << node->GetName();

            if (verbose_) {
                oss << " (OpType: " << node->GetOpType() << ", "
                    << "inputs: " << node->NumInputs() << ", "
                    << "outputs: " << node->NumOutputs() << ")";
            }
            oss << "\n";
        }
    }
    oss << std::string(60, '=') << "\n";

    // Print the output
    std::cout << oss.str() << std::flush;

    // Update context counter if available
    if (context) {
        context->IncrementCounter("nodes_printed", static_cast<int64_t>(nodes.size()));
    }

    return Status::Ok();
}

}  // namespace edgeunic
