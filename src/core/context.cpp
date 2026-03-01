#include "edgeunicompile/core/context.h"

namespace edgeunic {

std::shared_ptr<CompileContext> CompileContext::Create() {
    return std::make_shared<CompileContext>();
}

void CompileContext::SetOptLevel(int level) {
    if (level >= 0 && level <= 4) {
        opt_level_ = level;
    }
}

void CompileContext::SetSramSize(int64_t bytes) {
    if (bytes > 0) {
        sram_size_ = bytes;
    }
}

void CompileContext::SetWorkloadSize(int64_t bytes) {
    if (bytes >= 0) {
        workload_size_ = bytes;
    }
}

void CompileContext::SetTargetArch(const std::string& arch) {
    if (!arch.empty()) {
        target_arch_ = arch;
    }
}

void CompileContext::SetAttribute(const std::string& key, const AttributeValue& value) {
    attributes_[key] = value;
}

std::optional<AttributeValue> CompileContext::GetAttribute(const std::string& key) const {
    auto it = attributes_.find(key);
    if (it != attributes_.end()) {
        return it->second;
    }
    return std::nullopt;
}

void CompileContext::SetDebugMode(bool enabled) {
    debug_mode_ = enabled;
}

void CompileContext::SetVerboseMode(bool enabled) {
    verbose_mode_ = enabled;
}

void CompileContext::IncrementCounter(const std::string& name, int64_t value) {
    counters_[name] += value;
}

int64_t CompileContext::GetCounter(const std::string& name) const {
    auto it = counters_.find(name);
    if (it != counters_.end()) {
        return it->second;
    }
    return 0;
}

std::unordered_map<std::string, int64_t> CompileContext::GetAllCounters() const {
    return counters_;
}

void CompileContext::SetStatus(const Status& status) {
    status_ = status;
}

ContextManager& ContextManager::Instance() {
    static ContextManager instance;
    return instance;
}

std::shared_ptr<CompileContext> ContextManager::CreateContext() {
    auto context = std::make_shared<CompileContext>();
    SetCurrentContext(context);
    return context;
}

std::shared_ptr<CompileContext> ContextManager::GetCurrentContext() const {
    return current_context_;
}

void ContextManager::SetCurrentContext(std::shared_ptr<CompileContext> context) {
    current_context_ = std::move(context);
}

}  // namespace edgeunic
