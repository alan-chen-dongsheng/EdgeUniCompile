#pragma once

#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include "edgeunicompile/core/types.h"
#include "edgeunicompile/core/context.h"
#include "edgeunicompile/passes/pass_base.h"
#include "edgeunicompile/ir/graph.h"

namespace edgeunic {

/**
 * Manager for running multiple passes in sequence.
 *
 * The PassManager orchestrates the execution of passes, providing:
 * - Pass registration and ordering
 * - Pass configuration
 * - Execution pipeline
 */
class PassManager {
public:
    explicit PassManager(ContextPtr context = nullptr);
    ~PassManager() = default;

    /**
     * Add a pass to the manager.
     *
     * @param pass The pass to add.
     */
    void AddPass(PassPtr pass);

    /**
     * Remove a pass from the manager.
     *
     * @param pass_name Name of the pass to remove.
     */
    void RemovePass(const std::string& pass_name);

    /**
     * Get a pass by name.
     *
     * @param pass_name Name of the pass to get.
     * @return The pass instance or nullptr if not found.
     */
    PassPtr GetPass(const std::string& pass_name) const;

    /**
     * Run all enabled passes on the graph.
     *
     * @param graph The computation graph to optimize.
     * @return Status indicating success or failure.
     */
    Status Run(GraphPtr graph);

    /**
     * Run a specific pass by name.
     *
     * @param pass_name Name of the pass to run.
     * @param graph The computation graph to optimize.
     * @return Status indicating success or failure.
     */
    Status RunPass(const std::string& pass_name, GraphPtr graph);

    /**
     * Disable a pass by name.
     *
     * @param pass_name Name of the pass to disable.
     */
    void DisablePass(const std::string& pass_name);

    /**
     * Enable a pass by name.
     *
     * @param pass_name Name of the pass to enable.
     */
    void EnablePass(const std::string& pass_name);

    /**
     * List all registered passes with their status.
     *
     * @return Vector of pass information.
     */
    std::vector<std::pair<std::string, bool>> ListPasses() const;

    /**
     * Get the compilation context.
     */
    ContextPtr GetContext() const { return context_; }

    /**
     * Set the compilation context.
     */
    void SetContext(ContextPtr context) { context_ = context; }

private:
    ContextPtr context_;
    std::vector<PassPtr> passes_;
    std::unordered_map<std::string, PassPtr> pass_map_;
};

using PassManagerPtr = std::shared_ptr<PassManager>;

}  // namespace edgeunic
