#pragma once

#include <vector>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include "edgeunicompile/codegen/instruction.h"
#include "edgeunicompile/ir/graph.h"
#include "edgeunicompile/passes/pass_base.h"

namespace edgeunic {

/**
 * Instruction scheduler for generating and scheduling instructions.
 *
 * This class handles:
 * 1. DFS-based topological sort of nodes
 * 2. Generation of load/exec/store instructions for each node
 * 3. Dependency analysis between instructions
 * 4. Packet-based scheduling for parallel execution
 */
class InstructionScheduler {
public:
    InstructionScheduler();
    ~InstructionScheduler() = default;

    /**
     * Generate instructions from a graph.
     *
     * @param graph The computation graph.
     * @param context The compilation context.
     * @return Status indicating success or failure.
     */
    Status GenerateInstructions(const GraphPtr& graph, std::shared_ptr<PassContext> context = nullptr);

    /**
     * Schedule instructions into packets for parallel execution.
     *
     * Instructions with no dependencies are grouped into the same packet.
     *
     * @return Status indicating success or failure.
     */
    Status ScheduleInstructions();

    /**
     * Get the generated instruction packets.
     *
     * @return Vector of instruction packets in execution order.
     */
    const std::vector<InstructionPacketPtr>& GetPackets() const { return packets_; }

    /**
     * Get all generated instructions.
     *
     * @return Vector of all instructions in dependency order.
     */
    const std::vector<InstructionPtr>& GetInstructions() const { return instructions_; }

    /**
     * Print the instruction schedule.
     */
    void PrintSchedule() const;

    /**
     * Get node order from DFS topological sort.
     *
     * @return Vector of node names in DFS order.
     */
    const std::vector<std::string>& GetNodeOrder() const { return node_order_; }

private:
    /**
     * Perform DFS-based topological sort on the graph.
     *
     * @param graph The computation graph.
     */
    void TopologicalSortDFS(const GraphPtr& graph);

    /**
     * Generate load/exec/store instructions for a single node.
     *
     * @param node The node to generate instructions for.
     * @param graph The computation graph.
     */
    void GenerateNodeInstructions(const NodePtr& node, const GraphPtr& graph);

    /**
     * Build dependency graph between instructions.
     */
    void BuildInstructionDependencies();

    /**
     * Schedule instructions into packets using list scheduling.
     */
    void ListSchedule();

    // Node ordering from topological sort
    std::vector<std::string> node_order_;

    // All generated instructions
    std::vector<InstructionPtr> instructions_;

    // Instruction packets for parallel execution
    std::vector<InstructionPacketPtr> packets_;

    // Dependency graph: instruction index -> set of dependent instruction indices
    std::unordered_map<size_t, std::unordered_set<size_t>> dependencies_;

    // Reverse dependency graph: instruction index -> set of instructions it depends on
    std::unordered_map<size_t, std::unordered_set<size_t>> reverse_dependencies_;

    // Node name to ID mapping
    std::unordered_map<std::string, int> node_id_map_;

    // Tensor name to memory address mapping
    std::unordered_map<std::string, uint64_t> dram_addresses_;
    std::unordered_map<std::string, uint64_t> sram_addresses_;

    // Counters for address allocation
    uint64_t next_sram_address_ = 0;
    static constexpr uint64_t SRAM_BASE = 0;
    static constexpr uint64_t DRAM_BASE = 0;
    static constexpr uint64_t SRAM_MAX_SIZE = 3 * 1024 * 1024;  // 3MB
    static constexpr uint64_t DRAM_MAX_SIZE = 5ULL * 1024 * 1024 * 1024;  // 5GB
};

using InstructionSchedulerPtr = std::shared_ptr<InstructionScheduler>;

}  // namespace edgeunic
