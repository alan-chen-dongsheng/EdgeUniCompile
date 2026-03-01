#include "edgeunicompile/ir/tensor.h"

namespace edgeunic {

Tensor::Tensor(const std::string& name, DataType dtype, const Shape& shape, const std::string& producer_node)
    : name_(name), dtype_(dtype), shape_(shape), producer_node_(producer_node) {}

bool Tensor::IsValid() const {
    if (name_.empty()) {
        return false;
    }
    if (dtype_ == DataType::kUnknown) {
        return false;
    }
    if (!shape_.IsValid()) {
        return false;
    }
    if (!data_.empty()) {
        const size_t expected_size = shape_.NumElements() * GetDataTypeSize(dtype_);
        if (data_.size() != expected_size) {
            return false;
        }
    }
    return true;
}

std::string Tensor::ToString() const {
    std::string result = name_ + ": " + DataTypeToString(dtype_) + " " + shape_.ToString();
    if (!data_.empty()) {
        result += " (data size: " + std::to_string(data_.size()) + " bytes)";
    }
    if (!producer_node_.empty()) {
        result += " (produced by: " + producer_node_ + ")";
    }
    return result;
}

}  // namespace edgeunic
