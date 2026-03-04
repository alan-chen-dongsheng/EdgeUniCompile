#pragma once

#include "edgeunicompile/passes/pass_base.h"
#include "edgeunicompile/ir/graph.h"
#include <cstdint>

namespace edgeunic {

/**
 * Memory Allocation Pass using Linear Scan algorithm.
 *
 * This pass allocates memory for tensors in the computation graph
 * using a linear scan approach. It assigns memory offsets to tensors
 * based on their lifetime in the graph.
 *
 * Memory configuration:
 * - SRAM: Base address 0, Max size 3MB (3 * 1024 * 1024 bytes)
 * - DRAM: Base address 0, Max size 5GB (5 * 1024 * 1024 * 1024 bytes)
 */
class MemoryAllocationPass : public PassBase {
public:
    /**
     * Constructor.
     *
     * @param sram_base SRAM base address (default: 0)
     * @param sram_max_size SRAM maximum size (default: 3MB)
     * @param dram_base DRAM base address (default: 0)
     * @param dram_max_size DRAM maximum size (default: 5GB)
     */
    explicit MemoryAllocationPass(
        uint64_t sram_base = 0,
        uint64_t sram_max_size = 3 * 1024 * 1024,
        uint64_t dram_base = 0,
        uint64_t dram_max_size = 5UL * 1024 * 1024 * 1024
    );
    ~MemoryAllocationPass() override = default;

    Status Run(GraphPtr graph, std::shared_ptr<PassContext> context = nullptr) override;

    std::string GetDescription() const override;

private:
    /**
     * Calculate the size of a tensor in bytes.
     *
     * @param tensor The tensor to calculate size for.
     * @return Size in bytes.
     */
    size_t CalculateTensorSize(const Tensor& tensor) const;

    /**
     * Allocate memory for a single tensor.
     *
     * @param tensor The tensor to allocate.
     * @param is_sram Whether to allocate in SRAM.
     * @param current_offset Current offset in the memory region.
     * @return New offset after allocation.
     */
    uint64_t AllocateTensor(TensorPtr tensor, bool is_sram, uint64_t current_offset);

    uint64_t sram_base_;
    uint64_t sram_max_size_;
    uint64_t dram_base_;
    uint64_t dram_max_size_;

    uint64_t current_sram_offset_;
    uint64_t current_dram_offset_;
    uint64_t peak_sram_usage_;
    uint64_t peak_dram_usage_;
};

}  // namespace edgeunic
