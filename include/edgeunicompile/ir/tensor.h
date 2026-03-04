#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include "edgeunicompile/core/types.h"

namespace edgeunic {

// Memory location enum
enum class MemoryLocation : uint8_t {
    DRAM = 0,      // External DRAM (large, slow)
    SRAM = 1,      // On-chip SRAM (small, fast)
    L2Cache = 2,   // L2 cache (medium size, medium speed)
    Register = 3,  // Registers (very small, very fast)
};

class Tensor {
public:
    Tensor() = default;
    Tensor(const std::string& name, DataType dtype, const Shape& shape, const std::string& producer_node = "");

    std::string GetName() const { return name_; }
    void SetName(const std::string& name) { name_ = name; }

    DataType GetDataType() const { return dtype_; }
    void SetDataType(DataType dtype) { dtype_ = dtype; }

    const Shape& GetShape() const { return shape_; }
    void SetShape(const Shape& shape) { shape_ = shape; }

    size_t NumElements() const { return shape_.NumElements(); }

    const std::vector<uint8_t>& GetData() const { return data_; }
    std::vector<uint8_t>& GetData() { return data_; }

    void SetData(const std::vector<uint8_t>& data) { data_ = data; is_constant_ = !data_.empty(); }

    bool IsConstant() const { return is_constant_; }
    void SetIsConstant(bool is_constant) { is_constant_ = is_constant; }

    const std::string& GetProducerNode() const { return producer_node_; }
    void SetProducerNode(const std::string& node_name) { producer_node_ = node_name; }

    // Memory location and offset
    MemoryLocation GetMemoryLocation() const { return memory_location_; }
    void SetMemoryLocation(MemoryLocation location) { memory_location_ = location; }

    uint64_t GetMemoryOffset() const { return memory_offset_; }
    void SetMemoryOffset(uint64_t offset) { memory_offset_ = offset; }

    bool IsValid() const;
    std::string ToString() const;

private:
    std::string name_;
    DataType dtype_ = DataType::kUnknown;
    Shape shape_;
    std::vector<uint8_t> data_;
    bool is_constant_ = false;
    std::string producer_node_;
    MemoryLocation memory_location_ = MemoryLocation::DRAM;
    uint64_t memory_offset_ = 0;
};

using TensorPtr = std::shared_ptr<Tensor>;

}  // namespace edgeunic
