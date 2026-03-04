#include "edgeunicompile/codegen/instruction_scheduler.h"
#include <iostream>
#include <algorithm>
#include <stack>

namespace edgeunic {

InstructionScheduler::InstructionScheduler()
    : next_sram_address_(SRAM_BASE) {}

Status InstructionScheduler::GenerateInstructions(const GraphPtr& graph, std::shared_ptr<PassContext> context) {
    if (!graph) {
        return Status::InvalidArgument("Graph cannot be null");
    }

    // Clear previous state
    node_order_.clear();
    instructions_.clear();
    packets_.clear();
    dependencies_.clear();
    reverse_dependencies_.clear();
    node_id_map_.clear();
    dram_addresses_.clear();
    sram_addresses_.clear();
    next_sram_address_ = SRAM_BASE;

    // Assign node IDs
    int node_id = 0;
    for (const auto& node : graph->GetNodes()) {
        node_id_map_[node->GetName()] = node_id++;
    }

    // Perform DFS topological sort
    TopologicalSortDFS(graph);

    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << "DFS Node Order:\n";
    for (size_t i = 0; i < node_order_.size(); ++i) {
        std::cout << "  [" << i << "] " << node_order_[i] << "\n";
    }
    std::cout << std::string(60, '=') << "\n\n";

    // Generate instructions for each node in topological order
    for (const auto& node_name : node_order_) {
        auto node = graph->GetNode(node_name);
        if (node) {
            GenerateNodeInstructions(node, graph);
        }
    }

    // Build dependency graph
    BuildInstructionDependencies();

    // Update context if available
    if (context) {
        context->IncrementCounter("total_instructions", static_cast<int64_t>(instructions_.size()));
        context->IncrementCounter("total_nodes", static_cast<int64_t>(node_order_.size()));
    }

    return Status::Ok();
}

void InstructionScheduler::TopologicalSortDFS(const GraphPtr& graph) {
    std::unordered_set<std::string> visited;
    std::vector<std::string> result;

    // DFS helper function
    std::function<void(const std::string&)> dfs = [&](const std::string& node_name) {
        if (visited.count(node_name)) {
            return;
        }
        visited.insert(node_name);

        // Visit all dependent nodes first (nodes that produce inputs for this node)
        auto node = graph->GetNode(node_name);
        if (node) {
            // Find nodes that produce the inputs for this node
            for (const auto& input_tensor : node->GetInputs()) {
                for (const auto& other_node : graph->GetNodes()) {
                    for (const auto& output_tensor : other_node->GetOutputs()) {
                        if (output_tensor->GetName() == input_tensor->GetName()) {
                            dfs(other_node->GetName());
                            break;
                        }
                    }
                }
            }
        }

        result.push_back(node_name);
    };

    // Run DFS on all nodes
    for (const auto& node : graph->GetNodes()) {
        dfs(node->GetName());
    }

    node_order_ = std::move(result);
}

void InstructionScheduler::GenerateNodeInstructions(const NodePtr& node, const GraphPtr& graph) {
    int node_id = node_id_map_[node->GetName()];

    // Generate LOAD instructions for all inputs
    // This includes: IFM, weights, bias, gamma, beta, etc.
    for (const auto& input_tensor : node->GetInputs()) {
        auto load_instr = std::make_shared<LoadInstruction>();
        load_instr->SetName("load_" + input_tensor->GetName());
        load_instr->SetNodeId(node_id);

        // Set tensor name
        load_instr->SetTensorName(input_tensor->GetName());

        // Set size (float32 = 4 bytes)
        size_t size_bytes = input_tensor->GetShape().NumElements() * 4;
        load_instr->SetSizeBytes(size_bytes);

        // Set DRAM source address
        if (dram_addresses_.find(input_tensor->GetName()) == dram_addresses_.end()) {
            dram_addresses_[input_tensor->GetName()] = 0;  // Would need proper allocation
        }
        load_instr->SetSourceDramAddress(dram_addresses_[input_tensor->GetName()]);

        // Allocate SRAM address
        sram_addresses_[input_tensor->GetName()] = next_sram_address_;
        load_instr->SetTargetSramAddress(next_sram_address_);
        next_sram_address_ += size_bytes;

        instructions_.push_back(load_instr);
    }

    // Generate EXEC instruction
    auto exec_instr = std::make_shared<ExecInstruction>();
    exec_instr->SetName("exec_" + node->GetName());
    exec_instr->SetNodeId(node_id);

    // Set operation type - node->GetOpType() returns OpType enum
    exec_instr->SetOpType(node->GetOpType());

    // Set input SRAM addresses
    std::vector<uint64_t> input_addrs;
    for (const auto& input_tensor : node->GetInputs()) {
        input_addrs.push_back(sram_addresses_[input_tensor->GetName()]);
    }
    exec_instr->SetInputSramAddresses(input_addrs);

    // Allocate output SRAM address
    for (const auto& output_tensor : node->GetOutputs()) {
        size_t size_bytes = output_tensor->GetShape().NumElements() * 4;
        sram_addresses_[output_tensor->GetName()] = next_sram_address_;
        exec_instr->SetOutputSramAddress(next_sram_address_);
        next_sram_address_ += size_bytes;

        // Set DRAM address for store
        dram_addresses_[output_tensor->GetName()] = 0;  // Would need proper allocation
    }

    // Add attributes
    for (const auto& [key, value] : node->GetAttributes()) {
        exec_instr->AddAttribute(key, value);
    }

    instructions_.push_back(exec_instr);

    // Generate STORE instructions for all outputs
    for (const auto& output_tensor : node->GetOutputs()) {
        auto store_instr = std::make_shared<StoreInstruction>();
        store_instr->SetName("store_" + output_tensor->GetName());
        store_instr->SetNodeId(node_id);

        // Set tensor name
        store_instr->SetTensorName(output_tensor->GetName());

        // Set size
        size_t size_bytes = output_tensor->GetShape().NumElements() * 4;
        store_instr->SetSizeBytes(size_bytes);

        // Set SRAM source address
        store_instr->SetSourceSramAddress(sram_addresses_[output_tensor->GetName()]);

        // Set DRAM target address
        store_instr->SetTargetDramAddress(dram_addresses_[output_tensor->GetName()]);

        instructions_.push_back(store_instr);
    }
}

void InstructionScheduler::BuildInstructionDependencies() {
    // Build dependency graph based on node ordering and SRAM addresses
    // Rules:
    // 1. For each node: all LOADs must complete before EXEC
    // 2. EXEC must complete before all STOREs
    // 3. STORE must complete before any LOAD that uses the same tensor

    // Find instruction indices by node ID and type
    std::unordered_map<int, std::vector<size_t>> node_loads;
    std::unordered_map<int, size_t> node_exec;
    std::unordered_map<int, std::vector<size_t>> node_stores;

    for (size_t i = 0; i < instructions_.size(); ++i) {
        const auto& instr = instructions_[i];
        int node_id = instr->GetNodeId();

        switch (instr->GetType()) {
            case InstructionType::kLoad:
                node_loads[node_id].push_back(i);
                break;
            case InstructionType::kExec:
                node_exec[node_id] = i;
                break;
            case InstructionType::kStore:
                node_stores[node_id].push_back(i);
                break;
        }
    }

    // Add dependencies within each node
    for (const auto& [node_id, load_indices] : node_loads) {
        if (node_exec.find(node_id) == node_exec.end()) continue;

        size_t exec_idx = node_exec[node_id];

        // All loads -> exec
        for (size_t load_idx : load_indices) {
            dependencies_[load_idx].insert(exec_idx);
            reverse_dependencies_[exec_idx].insert(load_idx);
        }

        // Exec -> all stores
        if (node_stores.find(node_id) != node_stores.end()) {
            for (size_t store_idx : node_stores[node_id]) {
                dependencies_[exec_idx].insert(store_idx);
                reverse_dependencies_[store_idx].insert(exec_idx);
            }
        }
    }

    // Add dependencies across nodes based on tensor usage
    std::unordered_map<std::string, size_t> tensor_producer_store;

    for (size_t i = 0; i < instructions_.size(); ++i) {
        const auto& instr = instructions_[i];

        if (instr->GetType() == InstructionType::kStore) {
            auto store_instr = std::static_pointer_cast<StoreInstruction>(instr);
            tensor_producer_store[store_instr->GetTensorName()] = i;
        }
    }

    for (size_t i = 0; i < instructions_.size(); ++i) {
        const auto& instr = instructions_[i];

        if (instr->GetType() == InstructionType::kLoad) {
            auto load_instr = std::static_pointer_cast<LoadInstruction>(instr);
            const std::string& tensor_name = load_instr->GetTensorName();

            // This load depends on the store that produced this tensor
            if (tensor_producer_store.find(tensor_name) != tensor_producer_store.end()) {
                size_t producer_store_idx = tensor_producer_store[tensor_name];
                if (producer_store_idx != i) {  // Don't add self-dependency
                    dependencies_[producer_store_idx].insert(i);
                    reverse_dependencies_[i].insert(producer_store_idx);
                }
            }
        }
    }
}

Status InstructionScheduler::ScheduleInstructions() {
    if (instructions_.empty()) {
        return Status::Ok();
    }

    ListSchedule();

    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << "Instruction Schedule (" << packets_.size() << " packets)\n";
    std::cout << std::string(60, '=') << "\n";
    PrintSchedule();

    return Status::Ok();
}

void InstructionScheduler::ListSchedule() {
    packets_.clear();

    // Calculate in-degree for each instruction
    std::unordered_map<size_t, int> in_degree;
    for (size_t i = 0; i < instructions_.size(); ++i) {
        in_degree[i] = reverse_dependencies_[i].size();
    }

    // Track which instructions have been scheduled
    std::unordered_set<size_t> scheduled;

    // Schedule instructions level by level
    while (scheduled.size() < instructions_.size()) {
        // Find all ready instructions (in-degree = 0)
        std::vector<size_t> ready;
        for (size_t i = 0; i < instructions_.size(); ++i) {
            if (scheduled.count(i) == 0 && in_degree[i] == 0) {
                ready.push_back(i);
            }
        }

        if (ready.empty()) {
            // No ready instructions - might have a cycle or error
            std::cerr << "Warning: No ready instructions but not all scheduled!\n";
            break;
        }

        // Create a packet with all ready instructions
        auto packet = std::make_shared<InstructionPacket>();
        for (size_t idx : ready) {
            packet->AddInstruction(instructions_[idx]);
            scheduled.insert(idx);
        }
        packets_.push_back(packet);

        // Update in-degrees
        for (size_t idx : ready) {
            if (dependencies_.find(idx) != dependencies_.end()) {
                for (size_t dep_idx : dependencies_[idx]) {
                    in_degree[dep_idx]--;
                }
            }
        }
    }
}

void InstructionScheduler::PrintSchedule() const {
    for (size_t i = 0; i < packets_.size(); ++i) {
        std::cout << "\n--- Packet " << i << " ---\n";
        std::cout << packets_[i]->ToString();
    }
    std::cout << std::string(60, '=') << "\n";
}

}  // namespace edgeunic
