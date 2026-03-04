#include "edgeunicompile/passes/pass_manager.h"
#include <algorithm>

namespace edgeunic {

PassManager::PassManager(ContextPtr context) : context_(context) {}

void PassManager::AddPass(PassPtr pass) {
    if (!pass) {
        return;
    }
    passes_.push_back(pass);
    pass_map_[pass->GetName()] = pass;
}

void PassManager::RemovePass(const std::string& pass_name) {
    auto it = pass_map_.find(pass_name);
    if (it != pass_map_.end()) {
        passes_.erase(std::remove(passes_.begin(), passes_.end(), it->second), passes_.end());
        pass_map_.erase(it);
    }
}

PassPtr PassManager::GetPass(const std::string& pass_name) const {
    auto it = pass_map_.find(pass_name);
    if (it != pass_map_.end()) {
        return it->second;
    }
    return nullptr;
}

Status PassManager::Run(GraphPtr graph) {
    if (!graph) {
        return Status::InvalidArgument("Graph cannot be null");
    }

    for (const auto& pass : passes_) {
        if (!pass->IsEnabled()) {
            continue;
        }

        auto status = pass->Run(graph, nullptr);
        if (!status.IsOk()) {
            return Status::Error("Pass '" + pass->GetName() + "' failed: " + status.Message());
        }

        // Increment counter in context if available
        if (context_) {
            context_->IncrementCounter("pass_" + pass->GetName() + "_runs", 1);
        }
    }

    return Status::Ok();
}

Status PassManager::RunPass(const std::string& pass_name, GraphPtr graph) {
    auto pass = GetPass(pass_name);
    if (!pass) {
        return Status::NotFound("Pass '" + pass_name + "' not found");
    }

    if (!pass->IsEnabled()) {
        return Status::Error("Pass '" + pass_name + "' is not enabled");
    }

    auto status = pass->Run(graph, nullptr);
    if (!status.IsOk()) {
        return Status::Error("Pass '" + pass_name + "' failed: " + status.Message());
    }

    if (context_) {
        context_->IncrementCounter("pass_" + pass_name + "_runs", 1);
    }

    return Status::Ok();
}

void PassManager::DisablePass(const std::string& pass_name) {
    auto pass = GetPass(pass_name);
    if (pass) {
        pass->SetEnabled(false);
    }
}

void PassManager::EnablePass(const std::string& pass_name) {
    auto pass = GetPass(pass_name);
    if (pass) {
        pass->SetEnabled(true);
    }
}

std::vector<std::pair<std::string, bool>> PassManager::ListPasses() const {
    std::vector<std::pair<std::string, bool>> result;
    result.reserve(passes_.size());
    for (const auto& pass : passes_) {
        result.emplace_back(pass->GetName(), pass->IsEnabled());
    }
    return result;
}

}  // namespace edgeunic
