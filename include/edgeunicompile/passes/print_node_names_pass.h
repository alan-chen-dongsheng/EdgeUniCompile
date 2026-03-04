#pragma once

#include "edgeunicompile/passes/pass_base.h"
#include "edgeunicompile/ir/graph.h"

namespace edgeunic {

/**
 * Print Node Names Pass.
 *
 * This pass prints all node names in the computation graph.
 * Useful for debugging and understanding the graph structure.
 */
class PrintNodeNamesPass : public PassBase {
public:
    explicit PrintNodeNamesPass(bool verbose = false);
    ~PrintNodeNamesPass() override = default;

    Status Run(GraphPtr graph, std::shared_ptr<PassContext> context = nullptr) override;

    std::string GetDescription() const override;

private:
    bool verbose_;  // If true, print additional details
};

}  // namespace edgeunic
