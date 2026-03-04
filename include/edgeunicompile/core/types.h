#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <optional>
#include <variant>
#include <ostream>

namespace edgeunic {

// Data types supported by the compiler
enum class DataType {
    kUnknown = 0,
    kFloat32,
    kFloat16,
    kInt32,
    kInt16,
    kInt8,
    kUInt32,
    kUInt16,
    kUInt8,
    kBool,
    kComplex64,
};

// Get size of data type in bytes
size_t GetDataTypeSize(DataType type);

// Get string representation of data type
std::string DataTypeToString(DataType type);

// Parse string to data type
DataType StringToDataType(const std::string& str);

// Shape of a tensor (N-dimensional array)
struct Shape {
    std::vector<int64_t> dims;

    Shape() = default;
    Shape(std::initializer_list<int64_t> init) : dims(init) {}
    Shape(const std::vector<int64_t>& d) : dims(d) {}
    Shape(std::vector<int64_t>&& d) : dims(std::move(d)) {}

    bool operator==(const Shape& other) const { return dims == other.dims; }
    bool operator!=(const Shape& other) const { return !(*this == other); }

    size_t NumElements() const;
    size_t Rank() const { return dims.size(); }
    bool IsScalar() const { return dims.empty(); }
    bool IsValid() const;

    std::string ToString() const;
};

// Operator type
enum class OpType {
    kUnknown = 0,
    kAdd,
    kSubtract,
    kMultiply,
    kDivide,
    kConv2D,
    kMaxPool2D,
    kAveragePool2D,
    kRelu,
    kSigmoid,
    kTanh,
    kSoftmax,
    kMatMul,
    kReshape,
    kTranspose,
};

// Get string representation of operator type
std::string OpTypeToString(OpType type);

// Parse string to operator type
OpType StringToOpType(const std::string& str);

// Attribute value types
using AttributeValue = std::variant<
    std::monostate,
    bool,
    int64_t,
    float,
    double,
    std::string,
    std::vector<int64_t>,
    std::vector<float>,
    std::vector<double>,
    std::vector<std::string>,
    Shape
>;

// Status codes
enum class StatusCode {
    kOk = 0,
    kError,
    kInvalidArgument,
    kNotFound,
    kNotImplemented,
    kInternal,
    kResourceExhausted,
};

// Status object for error handling
class Status {
public:
    Status() : code_(StatusCode::kOk) {}
    Status(StatusCode code, const std::string& message) : code_(code), message_(message) {}

    static Status Ok() { return Status(); }
    static Status Error(const std::string& message) { return Status(StatusCode::kError, message); }
    static Status InvalidArgument(const std::string& message) { return Status(StatusCode::kInvalidArgument, message); }
    static Status NotFound(const std::string& message) { return Status(StatusCode::kNotFound, message); }
    static Status NotImplemented(const std::string& message) { return Status(StatusCode::kNotImplemented, message); }
    static Status Internal(const std::string& message) { return Status(StatusCode::kInternal, message); }
    static Status ResourceExhausted(const std::string& message) { return Status(StatusCode::kResourceExhausted, message); }

    bool IsOk() const { return code_ == StatusCode::kOk; }
    bool IsError() const { return !IsOk(); }

    StatusCode Code() const { return code_; }
    const std::string& Message() const { return message_; }

    std::string ToString() const;

private:
    StatusCode code_;
    std::string message_;
};

// Convenience macros for status checking
#define RETURN_IF_ERROR(status) \
    do { \
        const auto& _status = (status); \
        if (!_status.IsOk()) { \
            return _status; \
        } \
    } while (false)

#define ASSIGN_OR_RETURN(var, expr) \
    do { \
        auto _result = (expr); \
        if (!_result.status.IsOk()) { \
            return _result.status; \
        } \
        var = std::move(_result).value; \
    } while (false)

}  // namespace edgeunic

// Stream operators for debugging
std::ostream& operator<<(std::ostream& os, edgeunic::DataType type);
std::ostream& operator<<(std::ostream& os, edgeunic::Shape shape);
std::ostream& operator<<(std::ostream& os, edgeunic::OpType op);
std::ostream& operator<<(std::ostream& os, edgeunic::Status status);
