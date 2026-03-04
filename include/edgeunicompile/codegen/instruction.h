#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include "edgeunicompile/core/types.h"

namespace edgeunic {

// Forward declarations
class Tensor;
using TensorPtr = std::shared_ptr<Tensor>;

/**
 * Instruction type for code generation.
 */
enum class InstructionType {
    kLoad,   // Load data from DRAM to SRAM
    kExec,   // Execute computation
    kStore,  // Store result from SRAM to DRAM
};

/**
 * Convert instruction type to string.
 */
std::string InstructionTypeToString(InstructionType type);

/**
 * Memory location for instruction operands.
 */
enum class MemorySpace {
    kDRAM,   // External DRAM
    kSRAM,   // On-chip SRAM
};

/**
 * Base class for all instructions.
 */
class Instruction {
public:
    Instruction() = default;
    virtual ~Instruction() = default;

    InstructionType GetType() const { return type_; }
    const std::string& GetName() const { return name_; }
    void SetName(const std::string& name) { name_ = name; }
    int GetNodeId() const { return node_id_; }
    void SetNodeId(int id) { node_id_ = id; }

    virtual std::string ToString() const = 0;

protected:
    InstructionType type_ = InstructionType::kLoad;
    std::string name_;
    int node_id_ = -1;  // ID of the node this instruction belongs to
};

using InstructionPtr = std::shared_ptr<Instruction>;

/**
 * Load instruction - loads data from DRAM to SRAM.
 */
class LoadInstruction : public Instruction {
public:
    LoadInstruction();

    void SetTargetSramAddress(uint64_t addr) { target_sram_address_ = addr; }
    void SetSourceDramAddress(uint64_t addr) { source_dram_address_ = addr; }
    void SetSizeBytes(size_t size) { size_bytes_ = size; }
    void SetTensorName(const std::string& name) { tensor_name_ = name; }

    uint64_t GetTargetSramAddress() const { return target_sram_address_; }
    uint64_t GetSourceDramAddress() const { return source_dram_address_; }
    size_t GetSizeBytes() const { return size_bytes_; }
    const std::string& GetTensorName() const { return tensor_name_; }

    std::string ToString() const override;

private:
    uint64_t target_sram_address_ = 0;  // SRAM address to load to
    uint64_t source_dram_address_ = 0;  // DRAM address to load from
    size_t size_bytes_ = 0;              // Size of data to load
    std::string tensor_name_;            // Name of tensor being loaded
};

/**
 * Exec instruction - executes computation on SRAM data.
 */
class ExecInstruction : public Instruction {
public:
    ExecInstruction();

    void SetOpType(OpType op_type) { op_type_ = op_type; }
    void SetInputSramAddresses(const std::vector<uint64_t>& addrs) { input_sram_addresses_ = addrs; }
    void SetOutputSramAddress(uint64_t addr) { output_sram_address_ = addr; }
    void AddInputSramAddress(uint64_t addr) { input_sram_addresses_.push_back(addr); }
    void AddAttribute(const std::string& key, const AttributeValue& value) { attributes_[key] = value; }

    OpType GetOpType() const { return op_type_; }
    const std::vector<uint64_t>& GetInputSramAddresses() const { return input_sram_addresses_; }
    uint64_t GetOutputSramAddress() const { return output_sram_address_; }
    const std::unordered_map<std::string, AttributeValue>& GetAttributes() const { return attributes_; }

    std::string ToString() const override;

private:
    OpType op_type_ = OpType::kUnknown;
    std::vector<uint64_t> input_sram_addresses_;   // SRAM addresses of inputs
    uint64_t output_sram_address_ = 0;              // SRAM address of output
    std::unordered_map<std::string, AttributeValue> attributes_;  // Operation attributes
};

/**
 * Store instruction - stores data from SRAM to DRAM.
 */
class StoreInstruction : public Instruction {
public:
    StoreInstruction();

    void SetSourceSramAddress(uint64_t addr) { source_sram_address_ = addr; }
    void SetTargetDramAddress(uint64_t addr) { target_dram_address_ = addr; }
    void SetSizeBytes(size_t size) { size_bytes_ = size; }
    void SetTensorName(const std::string& name) { tensor_name_ = name; }

    uint64_t GetSourceSramAddress() const { return source_sram_address_; }
    uint64_t GetTargetDramAddress() const { return target_dram_address_; }
    size_t GetSizeBytes() const { return size_bytes_; }
    const std::string& GetTensorName() const { return tensor_name_; }

    std::string ToString() const override;

private:
    uint64_t source_sram_address_ = 0;   // SRAM address to store from
    uint64_t target_dram_address_ = 0;   // DRAM address to store to
    size_t size_bytes_ = 0;               // Size of data to store
    std::string tensor_name_;             // Name of tensor being stored
};

/**
 * Instruction packet - a group of instructions that can execute in parallel.
 *
 * Instructions in a packet have no dependencies on each other and can be
 * executed in parallel. Packets are executed in order.
 */
class InstructionPacket {
public:
    InstructionPacket() = default;

    void AddInstruction(const InstructionPtr& instr);
    const std::vector<InstructionPtr>& GetInstructions() const { return instructions_; }
    size_t NumInstructions() const { return instructions_.size(); }

    std::string ToString() const;

private:
    std::vector<InstructionPtr> instructions_;
};

using InstructionPacketPtr = std::shared_ptr<InstructionPacket>;

}  // namespace edgeunic
