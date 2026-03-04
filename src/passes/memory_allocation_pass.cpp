#include "edgeunicompile/passes/memory_allocation_pass.h"
#include "edgeunicompile/core/types.h"
#include <iostream>
#include <sstream>
#include <algorithm>

namespace edgeunic {

MemoryAllocationPass::MemoryAllocationPass(
    uint64_t sram_base,
    uint64_t sram_max_size,
    uint64_t dram_base,
    uint64_t dram_max_size
)
    : PassBase("memory_allocation_pass"),
      sram_base_(sram_base),
      sram_max_size_(sram_max_size),
      dram_base_(dram_base),
      dram_max_size_(dram_max_size),
      current_sram_offset_(0),
      current_dram_offset_(0),
      peak_sram_usage_(0),
      peak_dram_usage_(0) {}

std::string MemoryAllocationPass::GetDescription() const {
    return "MemoryAllocationPass: Allocates memory for tensors using linear scan algorithm";
}

size_t MemoryAllocationPass::CalculateTensorSize(const Tensor& tensor) const {
    // Calculate number of elements
    size_t num_elements = tensor.GetShape().NumElements();

    // Get data type size
    size_t element_size = GetDataTypeSize(tensor.GetDataType());

    return num_elements * element_size;
}

uint64_t MemoryAllocationPass::AllocateTensor(
    TensorPtr tensor,
    bool is_sram,
    uint64_t current_offset
) {
    if (!tensor) {
        return current_offset;
    }

    size_t tensor_size = CalculateTensorSize(*tensor);

    // Set memory location and offset
    if (is_sram) {
        tensor->SetMemoryLocation(MemoryLocation::SRAM);
        tensor->SetMemoryOffset(sram_base_ + current_offset);

        // Update SRAM offset
        uint64_t new_offset = current_offset + tensor_size;
        peak_sram_usage_ = std::max(peak_sram_usage_, new_offset);

        return new_offset;
    } else {
        tensor->SetMemoryLocation(MemoryLocation::DRAM);
        tensor->SetMemoryOffset(dram_base_ + current_offset);

        // Update DRAM offset
        uint64_t new_offset = current_offset + tensor_size;
        peak_dram_usage_ = std::max(peak_dram_usage_, new_offset);

        return new_offset;
    }
}

Status MemoryAllocationPass::Run(GraphPtr graph, std::shared_ptr<PassContext> context) {
    if (!graph) {
        return Status::InvalidArgument("Graph cannot be null");
    }

    // Reset counters
    current_sram_offset_ = 0;
    current_dram_offset_ = 0;
    peak_sram_usage_ = 0;
    peak_dram_usage_ = 0;

    std::ostringstream oss;
    oss << "\n" << std::string(60, '=') << "\n";
    oss << "Memory Allocation Report (Linear Scan)\n";
    oss << std::string(60, '-') << "\n";
    oss << "SRAM: Base=0x" << std::hex << sram_base_ << std::dec
        << ", Max=" << sram_max_size_ << " bytes ("
        << (sram_max_size_ / (1024 * 1024)) << " MB)\n";
    oss << "DRAM: Base=0x" << std::hex << dram_base_ << std::dec
        << ", Max=" << dram_max_size_ << " bytes ("
        << (dram_max_size_ / (1024 * 1024 * 1024)) << " GB)\n";
    oss << std::string(60, '-') << "\n";

    // Get all tensors
    const auto& tensors = graph->GetTensors();

    int sram_count = 0;
    int dram_count = 0;

    // Linear scan allocation
    // Strategy: Allocate constant tensors (weights) to DRAM first,
    // then allocate activations. Small activations go to SRAM, large ones to DRAM.

    for (const auto& tensor : tensors) {
        size_t tensor_size = CalculateTensorSize(*tensor);
        bool is_constant = tensor->IsConstant();

        // Determine memory location based on size and type
        // Constants (weights) always go to DRAM
        // Activations: small ones (< 64KB) to SRAM, large ones to DRAM
        const size_t SRAM_THRESHOLD = 64 * 1024;  // 64KB

        bool allocate_to_sram = !is_constant && (tensor_size <= SRAM_THRESHOLD);

        if (allocate_to_sram) {
            // Check if we have enough SRAM
            if (current_sram_offset_ + tensor_size <= sram_max_size_) {
                current_sram_offset_ = AllocateTensor(tensor, true, current_sram_offset_);
                sram_count++;
                oss << "  [SRAM] " << tensor->GetName()
                    << ": " << tensor_size << " bytes @ 0x"
                    << std::hex << (sram_base_ + tensor->GetMemoryOffset()) << std::dec
                    << " (" << (tensor_size / 1024.0) << " KB)\n";
            } else {
                // Fall back to DRAM
                allocate_to_sram = false;
            }
        }

        if (!allocate_to_sram) {
            // Allocate to DRAM
            if (current_dram_offset_ + tensor_size <= dram_max_size_) {
                current_dram_offset_ = AllocateTensor(tensor, false, current_dram_offset_);
                dram_count++;
                oss << "  [DRAM] " << tensor->GetName()
                    << ": " << tensor_size << " bytes @ 0x"
                    << std::hex << (dram_base_ + tensor->GetMemoryOffset()) << std::dec
                    << " (" << (tensor_size / 1024.0) << " KB)\n";
            } else {
                oss << "  [ERROR] " << tensor->GetName()
                    << ": Not enough DRAM (" << tensor_size << " bytes needed)\n";
                return Status::ResourceExhausted(
                    "DRAM exhausted. Needed: " + std::to_string(tensor_size) + " bytes"
                );
            }
        }
    }

    oss << std::string(60, '-') << "\n";
    oss << "Allocation Summary:\n";
    oss << "  SRAM tensors: " << sram_count
        << ", Peak usage: " << peak_sram_usage_ << " bytes ("
        << (peak_sram_usage_ / 1024.0) << " KB, "
        << (100.0 * peak_sram_usage_ / sram_max_size_) << "%)\n";
    oss << "  DRAM tensors: " << dram_count
        << ", Peak usage: " << peak_dram_usage_ << " bytes ("
        << (peak_dram_usage_ / 1024.0) << " KB, "
        << (100.0 * peak_dram_usage_ / dram_max_size_) << "%)\n";
    oss << "  Total tensors: " << tensors.size() << "\n";
    oss << std::string(60, '=') << "\n";

    std::cout << oss.str() << std::flush;

    // Update context if available
    if (context) {
        context->IncrementCounter("sram_tensors_allocated", sram_count);
        context->IncrementCounter("dram_tensors_allocated", dram_count);
        context->IncrementCounter("sram_peak_bytes", static_cast<int64_t>(peak_sram_usage_));
        context->IncrementCounter("dram_peak_bytes", static_cast<int64_t>(peak_dram_usage_));
    }

    return Status::Ok();
}

}  // namespace edgeunic
