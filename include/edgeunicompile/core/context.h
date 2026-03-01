#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include "types.h"

namespace edgeunic {

// Context for the compiler with configuration and state
class CompileContext {
public:
    static std::shared_ptr<CompileContext> Create();

    CompileContext() = default;
    ~CompileContext() = default;

    // Configuration options
    void SetOptLevel(int level);
    int GetOptLevel() const { return opt_level_; }

    void SetSramSize(int64_t bytes);
    int64_t GetSramSize() const { return sram_size_; }

    void SetWorkloadSize(int64_t bytes);
    int64_t GetWorkloadSize() const { return workload_size_; }

    void SetTargetArch(const std::string& arch);
    std::string GetTargetArch() const { return target_arch_; }

    void SetAttribute(const std::string& key, const AttributeValue& value);
    std::optional<AttributeValue> GetAttribute(const std::string& key) const;

    // Debug configuration
    void SetDebugMode(bool enabled);
    bool GetDebugMode() const { return debug_mode_; }

    void SetVerboseMode(bool enabled);
    bool GetVerboseMode() const { return verbose_mode_; }

    // Performance counters
    void IncrementCounter(const std::string& name, int64_t value = 1);
    int64_t GetCounter(const std::string& name) const;

    std::unordered_map<std::string, int64_t> GetAllCounters() const;

    // Status management
    void SetStatus(const Status& status);
    Status GetStatus() const { return status_; }
    bool IsOk() const { return status_.IsOk(); }

private:
    int opt_level_ = 3;
    int64_t sram_size_ = 32 * 1024 * 1024;  // 32MB default
    int64_t workload_size_ = 0;
    std::string target_arch_ = "armv8";
    bool debug_mode_ = false;
    bool verbose_mode_ = false;
    Status status_ = Status::Ok();
    std::unordered_map<std::string, AttributeValue> attributes_;
    std::unordered_map<std::string, int64_t> counters_;
};

// Global context manager
class ContextManager {
public:
    static ContextManager& Instance();

    std::shared_ptr<CompileContext> CreateContext();
    std::shared_ptr<CompileContext> GetCurrentContext() const;
    void SetCurrentContext(std::shared_ptr<CompileContext> context);

private:
    ContextManager() = default;
    ~ContextManager() = default;

    ContextManager(const ContextManager&) = delete;
    ContextManager& operator=(const ContextManager&) = delete;

    std::shared_ptr<CompileContext> current_context_;
};

}  // namespace edgeunic
