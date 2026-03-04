#pragma once

#include "edgeunicompile/passes/pass_base.h"
#include "edgeunicompile/ir/graph.h"

namespace edgeunic {

/**
 * Constant folding pass.
 *
 * This pass evaluates operations on constant tensors at compile time
 * and replaces them with the pre-computed results.
 */
class ConstantFoldingPass : public PassBase {
public:
    ConstantFoldingPass();
    ~ConstantFoldingPass() override = default;

    Status Run(GraphPtr graph, std::shared_ptr<PassContext> context = nullptr) override;

    std::string GetDescription() const override;

private:
    /**
     * Check if all inputs to a node are constants.
     */
    bool AreAllInputsConstant(const Node& node) const;

    /**
     * Fold a constant node.
     *
     * @param node The node to fold.
     * @param graph The graph containing the node.
     * @return Status indicating success or failure.
     */
    Status FoldNode(NodePtr node, GraphPtr graph);
};

}  // namespace edgeunic
