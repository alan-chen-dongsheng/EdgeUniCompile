// FFI Bindings Implementation
// This file provides C-compatible exports for calling C++ passes from Python

#include "edgeunicompile/ffi/ffi_bindings.h"
#include "edgeunicompile/flatbuf/flatbuf_converter.h"
#include "edgeunicompile/passes/print_node_names_pass.h"
#include "edgeunicompile/passes/memory_allocation_pass.h"
#include "edgeunicompile/passes/constant_folding_pass.h"
#include "edgeunicompile/passes/pass_manager.h"
#include <cstring>
#include <string>

// Helper function to create EUC_Result
static EUC_Result MakeResult(edgeunic::Status status, std::vector<uint8_t>* data = nullptr) {
    EUC_Result result;

    // Convert status code
    switch (status.Code()) {
        case edgeunic::StatusCode::kOk:
            result.code = EUC_OK;
            break;
        case edgeunic::StatusCode::kError:
            result.code = EUC_ERROR;
            break;
        case edgeunic::StatusCode::kInvalidArgument:
            result.code = EUC_INVALID_ARGUMENT;
            break;
        case edgeunic::StatusCode::kNotFound:
            result.code = EUC_NOT_FOUND;
            break;
        case edgeunic::StatusCode::kNotImplemented:
            result.code = EUC_NOT_IMPLEMENTED;
            break;
        case edgeunic::StatusCode::kInternal:
            result.code = EUC_INTERNAL;
            break;
        case edgeunic::StatusCode::kResourceExhausted:
            result.code = EUC_RESOURCE_EXHAUSTED;
            break;
    }

    // Copy message
    if (!status.Message().empty()) {
        result.message = new char[status.Message().size() + 1];
        std::strcpy(result.message, status.Message().c_str());
    } else {
        result.message = nullptr;
    }

    return result;
}

extern "C" {

EUC_Result EUC_RunPrintNodeNamesPass(
    const uint8_t* data,
    size_t size,
    int verbose
) {
    // Deserialize graph from FlatBuffer
    edgeunic::GraphPtr graph = edgeunic::flatbuf::FlatBufferConverter::Deserialize(data, size);

    if (!graph) {
        return MakeResult(edgeunic::Status::Internal("Failed to deserialize graph"));
    }

    // Create and run pass
    edgeunic::PrintNodeNamesPass pass(verbose != 0);
    auto run_status = pass.Run(graph, nullptr);

    if (!run_status.IsOk()) {
        return MakeResult(run_status);
    }

    // Note: The graph is modified in place
    // Python will re-parse the FlatBuffer if needed

    return MakeResult(edgeunic::Status::Ok());
}

EUC_Result EUC_RunMemoryAllocationPass(
    const uint8_t* data,
    size_t size,
    uint64_t sram_base,
    uint64_t sram_max_size,
    uint64_t dram_base,
    uint64_t dram_max_size
) {
    // Deserialize graph from FlatBuffer
    edgeunic::GraphPtr graph = edgeunic::flatbuf::FlatBufferConverter::Deserialize(data, size);

    if (!graph) {
        return MakeResult(edgeunic::Status::Internal("Failed to deserialize graph"));
    }

    // Create and run pass
    edgeunic::MemoryAllocationPass pass(
        sram_base,
        sram_max_size,
        dram_base,
        dram_max_size
    );
    auto run_status = pass.Run(graph, nullptr);

    if (!run_status.IsOk()) {
        return MakeResult(run_status);
    }

    // Note: The graph is modified in place
    // Python will re-parse the FlatBuffer if needed

    return MakeResult(edgeunic::Status::Ok());
}

EUC_Result EUC_RunConstantFoldingPass(
    const uint8_t* data,
    size_t size
) {
    // Deserialize graph from FlatBuffer
    edgeunic::GraphPtr graph = edgeunic::flatbuf::FlatBufferConverter::Deserialize(data, size);

    if (!graph) {
        return MakeResult(edgeunic::Status::Internal("Failed to deserialize graph"));
    }

    // Create and run pass
    edgeunic::ConstantFoldingPass pass;
    auto pass_context = std::make_shared<edgeunic::PassContext>();
    auto run_status = pass.Run(graph, pass_context);

    if (!run_status.IsOk()) {
        return MakeResult(run_status);
    }

    return MakeResult(edgeunic::Status::Ok());
}

void EUC_FreeString(char* str) {
    if (str) {
        delete[] str;
    }
}

void EUC_FreeGraphData(uint8_t* data) {
    if (data) {
        delete[] data;
    }
}

}  // extern "C"
