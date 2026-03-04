#include "edgeunicompile/codegen/instruction.h"
#include <sstream>
#include <iomanip>

namespace edgeunic {

std::string InstructionTypeToString(InstructionType type) {
    switch (type) {
        case InstructionType::kLoad:
            return "LOAD";
        case InstructionType::kExec:
            return "EXEC";
        case InstructionType::kStore:
            return "STORE";
        default:
            return "UNKNOWN";
    }
}

// LoadInstruction implementation
LoadInstruction::LoadInstruction() {
    type_ = InstructionType::kLoad;
}

std::string LoadInstruction::ToString() const {
    std::ostringstream oss;
    oss << "LOAD  " << std::left << std::setw(20) << tensor_name_
        << " DRAM[0x" << std::hex << source_dram_address_ << "]"
        << " -> SRAM[0x" << target_sram_address_ << "]"
        << " (" << std::dec << size_bytes_ << " bytes)";
    return oss.str();
}

// ExecInstruction implementation
ExecInstruction::ExecInstruction() {
    type_ = InstructionType::kExec;
}

std::string ExecInstruction::ToString() const {
    std::ostringstream oss;
    oss << "EXEC  " << OpTypeToString(op_type_)
        << " inputs:[";
    for (size_t i = 0; i < input_sram_addresses_.size(); ++i) {
        if (i > 0) oss << ", ";
        oss << "0x" << std::hex << input_sram_addresses_[i];
    }
    oss << "] -> SRAM[0x" << output_sram_address_ << "]" << std::dec;
    return oss.str();
}

// StoreInstruction implementation
StoreInstruction::StoreInstruction() {
    type_ = InstructionType::kStore;
}

std::string StoreInstruction::ToString() const {
    std::ostringstream oss;
    oss << "STORE " << std::left << std::setw(20) << tensor_name_
        << " SRAM[0x" << std::hex << source_sram_address_ << "]"
        << " -> DRAM[0x" << target_dram_address_ << "]"
        << " (" << std::dec << size_bytes_ << " bytes)";
    return oss.str();
}

// InstructionPacket implementation
void InstructionPacket::AddInstruction(const InstructionPtr& instr) {
    instructions_.push_back(instr);
}

std::string InstructionPacket::ToString() const {
    std::ostringstream oss;
    oss << "PACKET [" << instructions_.size() << " instructions]\n";
    for (const auto& instr : instructions_) {
        oss << "  " << instr->ToString() << "\n";
    }
    return oss.str();
}

}  // namespace edgeunic
