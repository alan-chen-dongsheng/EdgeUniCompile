#include "edgeunicompile/passes/pass_base.h"

namespace edgeunic {

// PassBase implementation
PassBase::PassBase(const std::string& name) : name_(name) {}

std::string PassBase::GetDescription() const {
    return name_;
}

// PassContext implementation
void PassContext::SetConfig(const std::string& key, const AttributeValue& value) {
    config_[key] = value;
}

std::optional<AttributeValue> PassContext::GetConfig(const std::string& key) const {
    auto it = config_.find(key);
    if (it != config_.end()) {
        return it->second;
    }
    return std::nullopt;
}

void PassContext::SetData(const std::string& key, const std::shared_ptr<void>& data) {
    data_[key] = data;
}

std::shared_ptr<void> PassContext::GetData(const std::string& key) const {
    auto it = data_.find(key);
    if (it != data_.end()) {
        return it->second;
    }
    return nullptr;
}

void PassContext::IncrementCounter(const std::string& key, int64_t delta) {
    counters_[key] += delta;
}

int64_t PassContext::GetCounter(const std::string& key) const {
    auto it = counters_.find(key);
    if (it != counters_.end()) {
        return it->second;
    }
    return 0;
}

}  // namespace edgeunic
