#include "edgeunicompile/core/types.h"

namespace edgeunic {

size_t GetDataTypeSize(DataType type) {
    switch (type) {
        case DataType::kFloat32:
            return 4;
        case DataType::kFloat16:
            return 2;
        case DataType::kInt32:
            return 4;
        case DataType::kInt16:
            return 2;
        case DataType::kInt8:
            return 1;
        case DataType::kUInt32:
            return 4;
        case DataType::kUInt16:
            return 2;
        case DataType::kUInt8:
            return 1;
        case DataType::kBool:
            return 1;
        case DataType::kComplex64:
            return 8;
        default:
            return 0;
    }
}

std::string DataTypeToString(DataType type) {
    switch (type) {
        case DataType::kFloat32:
            return "float32";
        case DataType::kFloat16:
            return "float16";
        case DataType::kInt32:
            return "int32";
        case DataType::kInt16:
            return "int16";
        case DataType::kInt8:
            return "int8";
        case DataType::kUInt32:
            return "uint32";
        case DataType::kUInt16:
            return "uint16";
        case DataType::kUInt8:
            return "uint8";
        case DataType::kBool:
            return "bool";
        case DataType::kComplex64:
            return "complex64";
        default:
            return "unknown";
    }
}

DataType StringToDataType(const std::string& str) {
    if (str == "float32") {
        return DataType::kFloat32;
    } else if (str == "float16") {
        return DataType::kFloat16;
    } else if (str == "int32") {
        return DataType::kInt32;
    } else if (str == "int16") {
        return DataType::kInt16;
    } else if (str == "int8") {
        return DataType::kInt8;
    } else if (str == "uint32") {
        return DataType::kUInt32;
    } else if (str == "uint16") {
        return DataType::kUInt16;
    } else if (str == "uint8") {
        return DataType::kUInt8;
    } else if (str == "bool") {
        return DataType::kBool;
    } else if (str == "complex64") {
        return DataType::kComplex64;
    }
    return DataType::kUnknown;
}

size_t Shape::NumElements() const {
    size_t num = 1;
    for (int64_t dim : dims) {
        num *= static_cast<size_t>(dim);
    }
    return num;
}

bool Shape::IsValid() const {
    for (int64_t dim : dims) {
        if (dim <= 0) {
            return false;
        }
    }
    return true;
}

std::string Shape::ToString() const {
    std::string str = "[";
    for (size_t i = 0; i < dims.size(); ++i) {
        if (i > 0) {
            str += ", ";
        }
        str += std::to_string(dims[i]);
    }
    str += "]";
    return str;
}

std::string OpTypeToString(OpType type) {
    switch (type) {
        case OpType::kAdd:
            return "Add";
        case OpType::kSubtract:
            return "Subtract";
        case OpType::kMultiply:
            return "Multiply";
        case OpType::kDivide:
            return "Divide";
        case OpType::kConv2D:
            return "Conv2D";
        case OpType::kMaxPool2D:
            return "MaxPool2D";
        case OpType::kAveragePool2D:
            return "AveragePool2D";
        case OpType::kRelu:
            return "Relu";
        case OpType::kSigmoid:
            return "Sigmoid";
        case OpType::kTanh:
            return "Tanh";
        case OpType::kSoftmax:
            return "Softmax";
        case OpType::kMatMul:
            return "MatMul";
        case OpType::kReshape:
            return "Reshape";
        case OpType::kTranspose:
            return "Transpose";
        case OpType::kEltwise:
            return "Eltwise";
        default:
            return "Unknown";
    }
}

OpType StringToOpType(const std::string& str) {
    if (str == "Add" || str == "add") {
        return OpType::kAdd;
    } else if (str == "Subtract" || str == "subtract" || str == "sub") {
        return OpType::kSubtract;
    } else if (str == "Multiply" || str == "multiply" || str == "mul") {
        return OpType::kMultiply;
    } else if (str == "Divide" || str == "divide" || str == "div") {
        return OpType::kDivide;
    } else if (str == "Conv2D" || str == "conv2d") {
        return OpType::kConv2D;
    } else if (str == "MaxPool2D" || str == "maxpool2d") {
        return OpType::kMaxPool2D;
    } else if (str == "AveragePool2D" || str == "averagepool2d") {
        return OpType::kAveragePool2D;
    } else if (str == "Relu" || str == "relu") {
        return OpType::kRelu;
    } else if (str == "Sigmoid" || str == "sigmoid") {
        return OpType::kSigmoid;
    } else if (str == "Tanh" || str == "tanh") {
        return OpType::kTanh;
    } else if (str == "Softmax" || str == "softmax") {
        return OpType::kSoftmax;
    } else if (str == "MatMul" || str == "matmul") {
        return OpType::kMatMul;
    } else if (str == "Reshape" || str == "reshape") {
        return OpType::kReshape;
    } else if (str == "Transpose" || str == "transpose") {
        return OpType::kTranspose;
    } else if (str == "Eltwise" || str == "eltwise") {
        return OpType::kEltwise;
    }
    return OpType::kUnknown;
}

std::string Status::ToString() const {
    switch (code_) {
        case StatusCode::kOk:
            return "OK";
        case StatusCode::kError:
            return "ERROR: " + message_;
        case StatusCode::kInvalidArgument:
            return "INVALID_ARGUMENT: " + message_;
        case StatusCode::kNotFound:
            return "NOT_FOUND: " + message_;
        case StatusCode::kNotImplemented:
            return "NOT_IMPLEMENTED: " + message_;
        case StatusCode::kInternal:
            return "INTERNAL: " + message_;
        case StatusCode::kResourceExhausted:
            return "RESOURCE_EXHAUSTED: " + message_;
    }
    return "UNKNOWN: " + message_;
}

}  // namespace edgeunic

// Stream operators for debugging
std::ostream& operator<<(std::ostream& os, edgeunic::DataType type) {
    os << edgeunic::DataTypeToString(type);
    return os;
}

std::ostream& operator<<(std::ostream& os, const edgeunic::Shape& shape) {
    os << shape.ToString();
    return os;
}

std::ostream& operator<<(std::ostream& os, edgeunic::OpType op) {
    os << edgeunic::OpTypeToString(op);
    return os;
}

std::ostream& operator<<(std::ostream& os, const edgeunic::Status& status) {
    os << status.ToString();
    return os;
}
