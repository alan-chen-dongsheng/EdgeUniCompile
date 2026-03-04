#pragma once

#include <memory>
#include <string>
#include <vector>
#include "edgeunicompile/core/types.h"
#include "edgeunicompile/ir/graph.h"

namespace edgeunic {

class PassContext;

/**
 * Base class for all passes in the EdgeUniCompile system.
 *
 * Passes can be implemented in C++ and are used to optimize,
 * transform, or analyze computation graphs.
 */
class PassBase {
public:
    explicit PassBase(const std::string& name);
    virtual ~PassBase() = default;

    // Get pass name
    const std::string& GetName() const { return name_; }

    // Check if pass is enabled
    bool IsEnabled() const { return enabled_; }
    void SetEnabled(bool enabled) { enabled_ = enabled; }

    /**
     * Run the pass on the given graph.
     *
     * @param graph The computation graph to process.
     * @param context Optional pass context for configuration.
     * @return Status indicating success or failure.
     */
    virtual Status Run(GraphPtr graph, std::shared_ptr<PassContext> context = nullptr) = 0;

    /**
     * Get a description of the pass.
     */
    virtual std::string GetDescription() const;

protected:
    std::string name_;
    bool enabled_ = true;
};

/**
 * Context for pass configuration and state.
 */
class PassContext {
public:
    PassContext() = default;

    // Set/get configuration values
    void SetConfig(const std::string& key, const AttributeValue& value);
    std::optional<AttributeValue> GetConfig(const std::string& key) const;

    // Set/get pass-specific data
    void SetData(const std::string& key, const std::shared_ptr<void>& data);
    std::shared_ptr<void> GetData(const std::string& key) const;

    // Increment counters
    void IncrementCounter(const std::string& key, int64_t delta = 1);
    int64_t GetCounter(const std::string& key) const;

    // Get all counters
    const std::unordered_map<std::string, int64_t>& GetCounters() const { return counters_; }

private:
    std::unordered_map<std::string, AttributeValue> config_;
    std::unordered_map<std::string, std::shared_ptr<void>> data_;
    std::unordered_map<std::string, int64_t> counters_;
};

using PassPtr = std::shared_ptr<PassBase>;
using PassContextPtr = std::shared_ptr<PassContext>;

}  // namespace edgeunic
